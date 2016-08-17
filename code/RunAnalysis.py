import SolarStormwatchIIAnalysis as ssw

def main():
    #ssw.match_all_classifications_to_ssw_events(active=True, latest=True)
    #username = "lukebarnard"
    #ssw.match_user_classifications_to_ssw_events(username, active=True, latest=True)
    ssw.test_plot()
    ssw.test_animation()
    ssw.test_front_reconstruction()
    return

if __name__ == "__main__":
    main()

