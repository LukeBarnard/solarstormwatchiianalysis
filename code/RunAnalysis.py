import SolarStormwatchIIAnalysis as ssw

def main():
    ssw.create_classification_frame_matched_hdf5(active=True, latest=True)
    ssw.test_plot()
    return

if __name__ == "__main__":
    main()

