from alpha_beta_filter import AlphaBetaFilter, DataLoader, DataAssociation
from cell_data_loader import CellDataLoader
from bat_data_loader import BatDataLoader

def exec_bat_self_gen_localization():
	bat_data = BatDataLoader("../data/bats/CS585-BatImages/FalseColor/", 2, DEBUG=False)
	bat_tracker = AlphaBetaFilter(bat_data, data_association_fn=DataAssociation.associate, window_size=(600,600), DEBUG=True)
	bat_tracker.run()

def exec_cell_self_gen_localization():
	data = CellDataLoader("../data/cell/CS585-Cells/", 2, DEBUG=False)
	cell_tracker = AlphaBetaFilter(data, data_association_fn=DataAssociation.associate, window_size=(600,600), DEBUG=True)
	cell_tracker.run()

if __name__ == "__main__":
	code_exec_choice = input("Choose an option below:\n\tEnter 1 for Bat Dataset (with self-generating localization)\n\tEnter 2 for Cell Dataset (with self-generating localization)\n")
	code_exec_h = {
		"1": exec_bat_self_gen_localization(),
		"2": exec_cell_self_gen_localization()
	}
	print("You chose {}".format(code_exec_choice))
	code_exec_h[code_exec_choice]()