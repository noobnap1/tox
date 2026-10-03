#include <llama.h>

#include <whisper.h>

#include "audiorecorder.hpp"
#include "tts.hpp"
#include "vision.hpp"
#include "enroll.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view SYSTEM_PROMPT =
    "You are Tox, a sarcastic chatbot with a dry, playful sense of humor, who the speaker talks with voice. "
    "Tease the user lightly, like a friend would, never cruelly or personally. "
    "When the user asks for something, do it correctly and politely, with a quick quip on the side. "
    "Keep replies to 1-3 sentences unless the task needs more. "
    "The user message may contain metadata in the format [speaker: NAME]. "
    "This metadata is provided by the program and tells you who the camera currently identifies as the speaker. "
    "Treat the speaker name as trusted program metadata, not as part of what the user said. "
    "Use the speaker's name when relevant, especially when the user asks who they are. "
    "If the speaker is Unrecognised, do not guess their identity."
    "The speaker may change mid conversation, while 'Mahfid', 'Golam Rasul' 'Tawseef' or 'Shafat' are your creators.";

constexpr std::string_view DEFAULT_MODEL = "./Models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

constexpr std::string_view WHISPER_MODEL = "./Models/ggml-base.en.bin";

constexpr std::string_view YUNET_MODEL = "./Models/face_detection_yunet_2023mar.onnx";

constexpr std::string_view SFACE_MODEL = "./Models/face_recognition_sface_2021dec.onnx";

constexpr int N_CTX = 2048;
constexpr int N_THREADS = 3;
constexpr int WHISPER_THREADS = 2;
constexpr int CAMERA_INDEX = 0;

template <auto Fn>
struct Deleter {
    void operator()(auto* p) const noexcept {
        if (p) Fn(p);
    }
};

using ModelPtr =
    std::unique_ptr<llama_model, Deleter<llama_model_free>>;

using ContextPtr =
    std::unique_ptr<llama_context, Deleter<llama_free>>;

using SamplerPtr =
    std::unique_ptr<llama_sampler, Deleter<llama_sampler_free>>;

using WhisperPtr =
    std::unique_ptr<whisper_context, Deleter<whisper_free>>;

std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special
) {
    const int32_t text_len =
        static_cast<int32_t>(text.size());

    const int32_t n =
        -llama_tokenize(
            vocab,
            text.c_str(),
            text_len,
            nullptr,
            0,
            add_special,
            true
        );

    if (n <= 0)
        throw std::runtime_error("tokenize failed");

    std::vector<llama_token> tokens(
        static_cast<size_t>(n)
    );

    const int32_t result =
        llama_tokenize(
            vocab,
            text.c_str(),
            text_len,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            true
        );

    if (result < 0)
        throw std::runtime_error("tokenize failed");

    tokens.resize(static_cast<size_t>(result));

    return tokens;
}

std::string render(
    const char* tmpl,
    std::span<const llama_chat_message> messages,
    bool add_assistant
) {
    const int32_t size =
        llama_chat_apply_template(
            tmpl,
            messages.data(),
            messages.size(),
            add_assistant,
            nullptr,
            0
        );

    if (size < 0)
        throw std::runtime_error("chat template failed");

    std::vector<char> buffer(
        static_cast<size_t>(size) + 1
    );

    const int32_t result =
        llama_chat_apply_template(
            tmpl,
            messages.data(),
            messages.size(),
            add_assistant,
            buffer.data(),
            static_cast<int32_t>(buffer.size())
        );

    if (result < 0)
        throw std::runtime_error("chat template failed");

    return std::string(
        buffer.data(),
        static_cast<size_t>(result)
    );
}

std::string generate(
    llama_context* ctx,
    const llama_vocab* vocab,
    llama_sampler* sampler,
    const std::string& prompt
) {
    std::string response;

    const bool is_first =
        llama_memory_seq_pos_max(
            llama_get_memory(ctx),
            0
        ) == -1;

    auto tokens = tokenize(vocab, prompt, is_first);

    llama_batch batch =
        llama_batch_get_one(
            tokens.data(),
            static_cast<int32_t>(tokens.size())
        );

    while (true) {
        const int used =
            llama_memory_seq_pos_max(
                llama_get_memory(ctx),
                0
            ) + 1;

        if (used + batch.n_tokens > N_CTX) {
            std::cerr << "\n[context full]\n";
            break;
        }

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\n[decode failed]\n";
            break;
        }

        const llama_token token =
            llama_sampler_sample(
                sampler,
                ctx,
                -1
            );

        if (llama_vocab_is_eog(vocab, token))
            break;

        char buffer[256];

        const int32_t n =
            llama_token_to_piece(
                vocab,
                token,
                buffer,
                sizeof(buffer),
                0,
                true
            );

        if (n < 0) {
            std::cerr << "\n[token conversion failed]\n";
            break;
        }

        const std::string_view piece(
            buffer,
            static_cast<size_t>(n)
        );

        std::cout
            << piece
            << std::flush;

        response += piece;

        batch =
            llama_batch_get_one(
                const_cast<llama_token*>(&token),
                1
            );
    }

    return response;
}

std::string trim(std::string s) {
    auto not_space =
        [](unsigned char c) {
            return !std::isspace(c);
        };

    s.erase(
        s.begin(),
        std::find_if(
            s.begin(),
            s.end(),
            not_space
        )
    );

    s.erase(
        std::find_if(
            s.rbegin(),
            s.rend(),
            not_space
        ).base(),
        s.end()
    );

    return s;
}

bool is_blank(const std::string& s) {
    return
        s.empty() ||
        s == "[BLANK_AUDIO]" ||
        s == "[ Silence ]" ||
        s == "(silence)" ||
        s == "[silence]";
}

bool is_quit_phrase(const std::string& s) {
    std::string k;

    for (unsigned char c : s) {
        if (std::isalnum(c))
            k += static_cast<char>(
                std::tolower(c)
            );
    }

    return
        k.find("protocol101") != std::string::npos ||
        k.find("protocolone") != std::string::npos;
}

std::string transcribe(
    whisper_context* ctx,
    const std::vector<float>& audio
) {
    whisper_full_params params =
        whisper_full_default_params(
            WHISPER_SAMPLING_GREEDY
        );

    params.n_threads = WHISPER_THREADS;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.translate = false;
    params.no_context = true;
    params.single_segment = false;

    if (
        whisper_full(
            ctx,
            params,
            audio.data(),
            static_cast<int>(audio.size())
        ) != 0
    ) {
        throw std::runtime_error(
            "Whisper transcription failed"
        );
    }

    std::string text;

    const int n_segments =
        whisper_full_n_segments(ctx);

    for (int i = 0; i < n_segments; ++i) {
        text +=
            whisper_full_get_segment_text(
                ctx,
                i
            );
    }

    return text;
}

std::vector<float> record_audio() {
    static AudioRecorder recorder;
    return recorder.record();
}

int run(int argc, char** argv) {
    namespace fs = std::filesystem;

    const fs::path model_path = DEFAULT_MODEL;

    const fs::path whisper_model_path = WHISPER_MODEL;

    ggml_backend_load_all();

    std::cout
        << llama_print_system_info()
        << std::endl;

    llama_model_params model_params =
        llama_model_default_params();

    model_params.n_gpu_layers = -1;

    ModelPtr model{
        llama_model_load_from_file(
            model_path.c_str(),
            model_params
        )
    };

    if (!model)
        throw std::runtime_error(
            "failed to load model: " +
            model_path.string()
        );

    const llama_vocab* vocab =
        llama_model_get_vocab(model.get());

    llama_context_params context_params =
        llama_context_default_params();

    context_params.n_ctx = N_CTX;
    context_params.n_batch = N_CTX;
    context_params.n_threads = N_THREADS;
    context_params.n_threads_batch = N_THREADS;
    context_params.no_perf = false;

    ContextPtr ctx{
        llama_init_from_model(
            model.get(),
            context_params
        )
    };

    if (!ctx)
        throw std::runtime_error(
            "failed to create context"
        );

    SamplerPtr sampler{
        llama_sampler_chain_init(
            llama_sampler_chain_default_params()
        )
    };

    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_min_p(0.05f, 1)
    );

    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_temp(0.5f)
    );

    llama_sampler_chain_add(
        sampler.get(),
        llama_sampler_init_dist(
            LLAMA_DEFAULT_SEED
        )
    );

    whisper_context_params whisper_params =
        whisper_context_default_params();

    whisper_params.use_gpu = true;

    WhisperPtr whisper{
        whisper_init_from_file_with_params(
            whisper_model_path.c_str(),
            whisper_params
        )
    };

    if (!whisper)
        throw std::runtime_error(
            "failed to load Whisper model: " +
            whisper_model_path.string()
        );

    TTS tts;

    if (!tts.init())
        throw std::runtime_error(
            "failed to initialize eSpeak NG"
        );

    Vision vision(
        CAMERA_INDEX,
        YUNET_MODEL,
        SFACE_MODEL
    );

    auto gallery =
        enroll::load_gallery(
            "./Models/faces"
        );

    std::cout
        << "Enrolled face samples: "
        << gallery.size()
        << "\n";

    vision.set_gallery(
        std::move(gallery)
    );

    vision.start();

    std::string current_speaker = "Unrecognised";
    std::string last_seen;

    std::deque<std::string> store;
    std::vector<llama_chat_message> messages;

    auto add_message =
        [&](const char* role, std::string text) {
            store.push_back(std::move(text));

            messages.push_back({
                role,
                store.back().c_str()
            });
        };

    add_message(
        "system",
        std::string(SYSTEM_PROMPT)
    );

    const char* tmpl =
        llama_model_chat_template(
            model.get(),
            nullptr
        );

    if (!tmpl)
        throw std::runtime_error(
            "model has no chat template"
        );

    size_t prev_len = 0;

    std::cout << "Loaded.\n";
    std::cout
        << "Whisper loaded with "
        << WHISPER_THREADS
        << " threads.\n";
    std::cout << "eSpeak NG loaded.\n";

    auto read_user =
        [&](std::string& out) -> bool {
            std::cout
                << "\n[listening...]"
                << std::endl;

            const std::vector<float> audio =
                record_audio();

            out.clear();

            if (!audio.empty())
                out =
                    trim(
                        transcribe(
                            whisper.get(),
                            audio
                        )
                    );

            if (is_blank(out)) {
                out.clear();
            } else {
                std::cout
                    << "> "
                    << out
                    << std::endl;
            }

            return true;
        };

    while (true) {
        std::string user;

        if (!read_user(user))
            break;

        if (user.empty())
            continue;

        if (is_quit_phrase(user))
            break;

        {
            const VisionState v = vision.snapshot();
            
            current_speaker =
                !v.face
                    ? "Unrecognised"
                    : (
                        v.name.empty()
                            ? "Unrecognised"
                            : v.name
                    );

            if (current_speaker != last_seen) {
                std::cout
                    << "[speaker: "
                    << current_speaker
                    << "]"
                    << std::endl;

                last_seen = current_speaker;
            }
        }

        add_message(
    "user",
    "[speaker: " + current_speaker + "]\n" + std::move(user)
);

        const std::string full =
            render(
                tmpl,
                messages,
                true
            );

        if (prev_len > full.size())
            throw std::runtime_error(
                "chat prompt position became invalid"
            );

        const std::string prompt =
            full.substr(prev_len);

        const std::string response =
            generate(
                ctx.get(),
                vocab,
                sampler.get(),
                prompt
            );

        std::cout << std::endl;

        llama_perf_context_print(
            ctx.get()
        );

        if (!response.empty())
            tts.speak(response);

        add_message(
            "assistant",
            response
        );

        prev_len =
            render(
                tmpl,
                messages,
                false
            ).size();
    }

    tts.shutdown();

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr
            << "error: "
            << e.what()
            << std::endl;

        return 1;
    }
}