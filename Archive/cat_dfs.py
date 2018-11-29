import pandas as pd
import os


def cat_dfs(ip_paths, idx_var = None, output = False, output_dsn = None):
	
	concat_dlist = []
	for ip_path in ip_paths:
		concat_dlist.append(pd.read_csv(ip_path, low_memory = False))
	concat_data = pd.concat([df for df in concat_dlist], ignore_index = True, sort = True)
	# Remove duplicates (may be some overlap)
	concat_data.drop_duplicates(keep = 'first', inplace = True) 	

	if output:

		ip_dir = os.path.dirname(ip_paths[0])
		concat_data.to_csv(
			os.path.join(ip_dir, output_dsn), 
			index = False, 
			encoding = 'utf-8'
		)
	
	return concat_data

ip_paths = [
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_5.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_6.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_7.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_8.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_9.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_10.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_11.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_12.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_1.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_2.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_3.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_4.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_5.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_6.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_7.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_8.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_9.txt',
	'/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2018_10.txt'
]
output_dsn = '/Users/josebolorinos/Google Drive/Codiga Center/lbre-cr2c-col.stanford.edu/Additional Parameters_2017_2018.csv'

cat_dfs(
	ip_paths,
	output = True,
	output_dsn = output_dsn
)