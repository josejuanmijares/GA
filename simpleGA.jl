include("reinit_DummyTest.jl")

function simpleGA(g, QA_LEVEL)
	g.par_evaluateAll();
	id_best, score_best =g.getBest()
	k_iteration = 0;
	while score_best > QA_LEVEL
		parent_a, parent_b = g.randomSelection(g.N)
		g.orderedCrossOver(parent_a, parent_b)
		g.exchangeMutation(parent_a, parent_b, 0.1)
		g.par_evaluateAll()
		id_best, score_best =g.getBest()
		println(" iter = $(k_iteration)    score = $(score_best)")
		k_iteration += 1;
	end
end


function singlePopulationGA()
	Npopulations = 2
	Nnumbers = 32
	Nbins = 16
	
	g = GAmodels(Npopulations,Nnumbers,Nbins);
	QA_LEVEL = 1.0

	#call simple GA
	simpleGA(g, QA_LEVEL)

end


function main()
	
	singlePopulationGA()
	
	# Npopulations = 128
	# Nnumbers = 32
	# Nbins = 16
	#
	# g = GAmodels(Npopulations,Nnumbers,Nbins);
	# g.par_evaluateAll();
	#
	# QA_LEVEL = 1.0
	#
	# #call simple GA
	# simpleGA(g, QA_LEVEL)
end

main()

