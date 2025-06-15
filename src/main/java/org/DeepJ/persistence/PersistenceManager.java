package org.DeepJ.persistence;

import org.DeepJ.ann.Network;
import java.io.*;

public class PersistenceManager {

    private static void serializeObject(Object object, String filePath) {
        try {
            FileOutputStream fileOut = new FileOutputStream(filePath);
            ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
            objectOut.writeObject(object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Object deserializeObject(String filePath) {
        try {
            FileInputStream fileIn = new FileInputStream(filePath);
            ObjectInputStream objectIn = new ObjectInputStream(fileIn);
            return objectIn.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void saveNetwork(Object network, String filePath) {
        serializeObject(network, filePath);
    }

    public static Network loadNetwork(String filePath) {
        return (Network) deserializeObject(filePath);
    }
}
