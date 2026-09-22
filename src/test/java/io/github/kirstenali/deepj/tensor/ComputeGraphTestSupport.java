package io.github.kirstenali.deepj.tensor;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

abstract class ComputeGraphTestSupport {

    protected RecordingRuntime runtime;
    protected ComputeGraph graph;

    @BeforeEach
    void setUp() {
        runtime = new RecordingRuntime();
        graph = new ComputeGraph(runtime);
    }

    static class RecordingRuntime implements GpuRuntime {

        record AllocCall(int[] ids, int[] sizes, int count) {}
        record UploadCall(int bufId, float[] data) {}
        record DownloadCall(int bufId, float[] output) {}
        record FlushCall(int[] cmdStream, int cmdStreamLength) {}
        record ReleaseCall(int[] ids, int count) {}

        final List<AllocCall> allocCalls = new ArrayList<>();
        final List<UploadCall> uploads = new ArrayList<>();
        final List<DownloadCall> downloads = new ArrayList<>();
        final List<FlushCall> flushCalls = new ArrayList<>();
        final List<ReleaseCall> releaseCalls = new ArrayList<>();

        float[] downloadResult = null;

        @Override
        public void allocBuffers(int[] ids, int[] sizes, int count) {
            allocCalls.add(new AllocCall(ids.clone(), sizes.clone(), count));
        }

        @Override
        public void uploadBuffer(int bufId, float[] data) {
            uploads.add(new UploadCall(bufId, data.clone()));
        }

        @Override
        public void downloadBuffer(int bufId, float[] out) {
            downloads.add(new DownloadCall(bufId, out));
            if (downloadResult != null) {
                System.arraycopy(downloadResult, 0, out, 0,
                        Math.min(downloadResult.length, out.length));
            }
        }

        @Override
        public void releaseBuffers(int[] ids, int count) {
            releaseCalls.add(new ReleaseCall(ids.clone(), count));
        }

        @Override
        public void flushOps(int[] cmdStream, int cmdStreamLength) {
            flushCalls.add(new FlushCall(cmdStream.clone(), cmdStreamLength));
        }
    }
}
