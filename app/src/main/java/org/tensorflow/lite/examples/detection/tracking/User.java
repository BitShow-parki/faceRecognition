package org.tensorflow.lite.examples.detection.tracking;

import java.util.List;
public class User {
    String name;
    int HR;

    public String getName() {
        return name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public int getAge() {
        return HR;
    }
    public void setAge(int heartrate) {
        this.HR = heartrate;
    }

    @Override
    public String toString() {
        return "User{" +
                "name='" + name + '\'' +
                ", Heartrate=" + HR +
                '}';
    }
}