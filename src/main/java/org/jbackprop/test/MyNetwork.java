package org.jbackprop.test;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.ann.Network;

public class MyNetwork extends Network {
    public MyNetwork(DataSet dataSet, int... neuronLayout) {
        super(dataSet, neuronLayout);
    }

    @Override
    public void afterEpoch() {
        super.afterEpoch();
        System.out.println("Total Loss: " + getLossOfEpoch());
    }
}
