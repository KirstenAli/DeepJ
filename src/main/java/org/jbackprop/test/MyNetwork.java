package org.jbackprop.test;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.ann.Network;
import org.jbackprop.ann.NetworkParams;

public class MyNetwork extends Network {
    public MyNetwork(NetworkParams networkParams, DataSet dataSet, int... neuronLayout) {
        super(networkParams, dataSet, neuronLayout);
    }

    public MyNetwork(DataSet dataSet, int... neuronLayout) {
        super(dataSet, neuronLayout);
    }

    @Override
    public void afterEpoch() {
        super.afterEpoch();
        System.out.println("Total Loss: " + getLossOfEpoch());
    }
}
