from __future__ import annotations


def edit_legend_labels(labels: list[str]) -> list[str]:
    """Edit the legend labels to be more readable."""
    new_labels = []
    for label in labels:
        _label = label
        if "_w_continuations" in label:
            _label = label.replace("_w_continuations", "")
        if ";priors=" in label:
            _label = label.replace(";priors=", "_")
        new_labels.append(_label)
    return new_labels



reference_points_dict = {

    # PD1
    "pd1-cifar100-wide_resnet-2048": {"valid_error_rate": 1, "train_cost": 100},
    "pd1-imagenet-resnet-512": {"valid_error_rate": 1, "train_cost": 5000},
    "pd1-lm1b-transformer-2048": {"valid_error_rate": 1, "train_cost": 1000},
    "pd1-translate_wmt-xformer_translate-64": {"valid_error_rate": 1, "train_cost": 20000},

    # JAHSBench
    "jahs-CIFAR10": {"valid_acc": 0, "runtime": 200000},
    "jahs-ColorectalHistology": {"valid_acc": 0, "runtime": 200000},
    "jahs-FashionMNIST": {"valid_acc": 0, "runtime": 200000},

    # MOMFPark
    "MOMFPark": {"value1": 1, "value2": 1},

    # YAHPO-LCBench
    "yahpo-lcbench-126026": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-167190": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 200
    },
    "yahpo-lcbench-168330": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 20000
    },
    "yahpo-lcbench-168910": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189906": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 1000
    },
    "yahpo-lcbench-3945": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-7593": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-34539": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126025": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-126029": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-146212": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167104": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167149": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167152": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167161": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167168": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167181": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167184": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167185": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167200": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-167201": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168329": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168331": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168335": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168868": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-168908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189354": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189862": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189865": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189866": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189873": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189905": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189908": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    },
    "yahpo-lcbench-189909": {
        "val_accuracy": 0,
        "val_balanced_accuracy": 0,
        "val_cross_entropy": 1,
        "time": 10000
    }
}