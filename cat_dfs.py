import pandas as pd


def cat_dfs(ip_paths, idx_var = None, output = False):
	
	concat_dlist = []
	for ip_path in ip_paths:
		concat_dlist.append(pd.read_csv(ip_path))
	concat_data = pd.concat([df for df in concat_dlist], ignore_index = True)
	# Remove duplicates (may be some overlap)
	concat_data.drop_duplicates(keep = 'first', inplace = True)
	
	# Sort by index (if given)
	if idx_var:
		concat_data.sort_values(idx_var, inplace = True)

	if output:
		ip_dir = os.path.dirname(ip_paths[0])
		concat_data.to_csv(ip_dir, index = False, encoding = 'utf-8')
	
	return concat_data