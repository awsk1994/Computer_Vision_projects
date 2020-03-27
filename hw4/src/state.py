class State:
	def __init__(self):
		"""
		x, y = centroid
		w, h = bounding box width and height
		vx, vy = velocity of x and y
		vw, vh = velocity of width and height
		"""
		self.x, self.y, self.w, self.h = 0, 0, 0, 0
		self.vx, self.vy, self.vw, self.vh = 0, 0, 0, 0

	def get_centroid(self):
		return [self.x, self.y]

	def set_centroid(self, cx, cy):
		self.x, self.y = cx, cy

	def get_bbox(self):
		return w, h

	def set_bbox(self, w, h):
		self.w, self.h = w, h

	def get_velocity_xy(self):
		return [self.vx, self.vy]

	def set_velocity_xy(self, vx, vy):
		self.vx, self.vy = vx, vy

	def get_velocity_bbox(self):
		return [self.vw, self.vh]

	def set_velocity_bbox(self, vw, vh):
		self.vw, self.vh = vw, vh

	def to_array(self):
		return [self.x, self.y, self.w, self.h, self.vx, self.vy, self.vw, self.vh]

	def update_from_arr(self, lst):
		self.x, self.y, self.w, self.h, self.vx, self.vy, self.vw, self.vh = lst
