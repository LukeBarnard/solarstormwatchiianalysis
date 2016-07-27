import SolarStormwatchIIAnalysis as ssw

def main():
    data = ssw.import_classifications(latest=True, version=27.17)
    print data
    return

if __name__ == "__main__":
    main()

