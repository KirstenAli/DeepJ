package io.github.kirstenali.deepj;

import io.github.kirstenali.deepj.tensor.Tensor;

import java.io.InputStream;
import java.lang.reflect.Method;
import java.net.URL;
import java.net.URLClassLoader;

/** Loads real DeepJ classes without a native library, independently of the host GPU. */
public final class MetalUnavailableClassLoader extends URLClassLoader {

    public MetalUnavailableClassLoader() {
        super(new URL[]{Tensor.class.getProtectionDomain().getCodeSource().getLocation()},
                ClassLoader.getPlatformClassLoader());
    }

    @Override
    public InputStream getResourceAsStream(String name) {
        if (name.equals("native/macos/libdeepj_metal_jni.dylib")) return null;
        return super.getResourceAsStream(name);
    }

    public Method method(String type, String name, Class<?>... parameters) throws ReflectiveOperationException {
        Method method = loadClass("io.github.kirstenali.deepj." + type).getDeclaredMethod(name, parameters);
        method.setAccessible(true);
        return method;
    }
}
