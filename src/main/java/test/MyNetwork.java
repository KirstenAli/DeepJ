package test;

import org.jbackprop.ann.Network;

public class MyNetwork extends Network {

    @Override
    public void afterEpoch() {
        super.afterEpoch();
        System.out.println("Loss of epoch " + getCurrentEpoch() + ": " + getLossOfEpoch());
    }
}
