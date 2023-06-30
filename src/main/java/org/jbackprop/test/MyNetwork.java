package org.jbackprop.test;

import org.jbackprop.dataset.DataSet;
import org.jbackprop.ann.Network;
import org.jbackprop.ann.NetworkParams;

public class MyNetwork extends Network {
    public MyNetwork(NetworkParams networkParams, DataSet dataSet, int... neuronLayout) {
        super(networkParams, dataSet, neuronLayout);
    }

    @Override
    public void afterEpoch() {
        super.afterEpoch();
    }
}
