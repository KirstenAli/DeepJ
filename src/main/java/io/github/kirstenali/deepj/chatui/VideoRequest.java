package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import io.github.kirstenali.deepj.persistence.qwen3.Qwen3Json;
import java.util.Map;
import io.github.kirstenali.deepj.tensor.DType;

record VideoRequest(SanaVideo.Options options, int fps) {
    VideoRequest {
        if (fps < 1 || fps > 100) throw new IllegalArgumentException("FPS must be from 1 to 100");
    }

    static VideoRequest read(String json) {
        Map<String, Object> data = Qwen3Json.object(Qwen3Json.parse(json));
        var options = new SanaVideo.Options(Qwen3Json.integer(data, "width"), Qwen3Json.integer(data, "height"),
                Qwen3Json.integer(data, "frames"), Qwen3Json.integer(data, "steps"),
                number(data, "guidance").floatValue(), Qwen3Json.integer(data.get("seed")),
                DType.valueOf(Qwen3Json.string(data.getOrDefault("dtype", "FLOAT32"))
                        .toUpperCase(java.util.Locale.ROOT)));
        return new VideoRequest(options, Qwen3Json.integer(data, "fps"));
    }

    private static Number number(Map<String, Object> data, String key) {
        if (!(data.get(key) instanceof Number value)) throw new IllegalArgumentException("Missing video " + key);
        if (!Double.isFinite(value.doubleValue())) throw new IllegalArgumentException("Invalid video " + key);
        return value;
    }
}
