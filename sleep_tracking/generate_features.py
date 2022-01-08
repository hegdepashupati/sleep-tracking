from sleep_tracking.features.psg_features import PSGLabels
from sleep_tracking.features.heartrate_features import HeartRateFeatures
from sleep_tracking.features.motion_features import MotionFeatures
from sleep_tracking.features.time_features import TimeFeatures
from sleep_tracking.context import Context


# main function to generate all the features
def main():
    print("Generating PSG labels ...")
    feature_loader = PSGLabels()
    feature_loader.write(Context.SUBJECTS)

    print("Generating heart rate features ...")
    feature_loader = HeartRateFeatures()
    feature_loader.write(Context.SUBJECTS)

    print("Generating motion features ...")
    feature_loader = MotionFeatures()
    feature_loader.write(Context.SUBJECTS)

    print("Generating time-based features ...")
    feature_loader = TimeFeatures()
    feature_loader.write(Context.SUBJECTS)

    print("Completed generating features.")


# entry point of the script
if __name__ == "__main__":
    main()

