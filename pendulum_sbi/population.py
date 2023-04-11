import yaml
import os

config_dir = os.path.join(os.path.dirname(__file__), '.')

class Population:
	def __init__(self):
		pass

	def load_yaml_file(self, file_name):
		"""Loads all the configs from the yaml file"""
		config_file = os.path.join(config_dir, file_name)
		with open (config_file, 'r') as config_file_obj:
			config_dict = yaml.safe_load(config_file_obj)
		return config_dict

	# You can also have other setup stuff in here like stuff that assigns
	# values if not assigned by the user


