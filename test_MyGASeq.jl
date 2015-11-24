
push!(LOAD_PATH, pwd()); 
LOAD_PATH = unique(LOAD_PATH);
workspace();
reload("MyGASeq")
using MyGASeq



function main()

	N_random_num = 1024
	N_window = 32
	
	N_bins = N_window/2
	 
	

end


# call main
main()