from bat_data_loader import BatDataLoader

bat_data_loader = BatDataLoader("../data/bats/CS585-BatImages/", 2, DEBUG=True)
bat_locs_all_frames = bat_data_loader.localization

print("len = {}".format(len(bat_locs_all_frames)))


bat_locs_thru_frames = []

for frame_idx in range(len(bat_locs_all_frames)):
    bat_locs_per_frame = bat_locs_all_frames[frame_idx]
    bat_locs = []
    for bat_loc in bat_locs_per_frame:
        [loc_x, loc_y] = bat_loc.get_centroid()
        bat_locs.append("{},{}".format(int(loc_x)*2, int(loc_y)*2))
    bat_locs_thru_frames.append(bat_locs)

# print(bat_locs_thru_frames)


for frame_idx in range(len(bat_locs_thru_frames)):
    f = open("../data/bats/Localization3/{}.txt".format(700 + frame_idx), "w")
    for bat_loc in bat_locs_thru_frames[frame_idx]:
        f.write(bat_loc)
        f.write("\n")
    f.close()