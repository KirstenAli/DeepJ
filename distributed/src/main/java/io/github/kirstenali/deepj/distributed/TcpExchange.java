package io.github.kirstenali.deepj.distributed;

import java.io.IOException;

interface TcpExchange extends AutoCloseable {
    byte[] exchange(long sequence, int operation, String key, byte[] bytes) throws IOException;
    @Override void close();
}
