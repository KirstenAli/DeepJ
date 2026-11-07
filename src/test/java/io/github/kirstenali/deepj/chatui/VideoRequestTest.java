package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import org.junit.jupiter.api.Test;
import io.github.kirstenali.deepj.tensor.DType;
import static org.junit.jupiter.api.Assertions.*;

class VideoRequestTest {
    private static final String SETTINGS = "{\"width\":128,\"height\":96,\"frames\":9,\"steps\":1,"
            + "\"seed\":42,\"fps\":16,\"guidance\":6";

    @Test void requestsKeepFloat32ByDefaultAndAcceptExplicitCompactPrecision() {
        assertEquals(new SanaVideo.Options(128, 96, 9, 1, 6, 42), VideoRequest.read(SETTINGS + "}").options());
        for (String dtype : new String[]{"bfloat16", "float32"}) {
            assertEquals(DType.valueOf(dtype.toUpperCase(java.util.Locale.ROOT)),
                    VideoRequest.read(SETTINGS + ",\"dtype\":\"" + dtype + "\"}").options().dtype());
        }
    }
}
