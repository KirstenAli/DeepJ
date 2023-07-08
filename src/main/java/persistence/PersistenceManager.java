package persistence;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;


public class PersistenceManager {

    public static void saveObjectAsJson(Object object, String filePath, String fileName) {
        // Create an ObjectMapper instance
        ObjectMapper objectMapper = new ObjectMapper();

        try {
            // Convert the object to JSON and write it to a file
            objectMapper.writeValue(new File(filePath + fileName), object);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
