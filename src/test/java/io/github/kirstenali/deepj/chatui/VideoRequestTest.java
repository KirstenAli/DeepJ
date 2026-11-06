package io.github.kirstenali.deepj.chatui;

import io.github.kirstenali.deepj.nn.diffusion.sanavideo.SanaVideo;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class VideoRequestTest {
    private static final String SETTINGS = "{\"width\":128,\"height\":96,\"frames\":9,\"steps\":1,"
            + "\"seed\":42,\"fps\":16,\"guidance\":6";

    @Test void requestsUseTheFloat32PipelineAndRejectRemovedPrecisionSettings() {
        assertEquals(new SanaVideo.Options(128, 96, 9, 1, 6, 42), VideoRequest.read(SETTINGS + "}").options());
        for (String dtype : new String[]{"bfloat16", "float32"}) {
            assertThrows(IllegalArgumentException.class,
                    () -> VideoRequest.read(SETTINGS + ",\"dtype\":\"" + dtype + "\"}"));
        }
    }
}
