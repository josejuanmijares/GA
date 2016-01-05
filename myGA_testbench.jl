if any(LOAD_PATH .== pwd())==false
	include("init_myGA.jl")
else
	include("reinit_myGA.jl")
end
		# rouletteWheelSelection				rWSel
		# rankSelection									rSel
		# tournamentSelection						tSel
		#
		# positionBasedCrossOver				pBXO
		# orderOneCrossOver							oOXO
		# partialMatchCrossOver					pMXO
		# cycleCrossOver								cXO
		#
		# displacementMutation					dM
		# scrambleMutation							sM
		# inversionMutation							iM
		# swapMutation									sM
		# exchangeMutation							xM
		# insertMutation								iM

		# replaceWorst									rW
		# elitism												e
		# roundRobinTournament					rRT
		# muReplacement									mR

function test1(g::GAmodel, fitness_value::Float64)
	g.evaluateAll();	
	k = 1
	while g.getBest()[1] > fitness_value
		parentA, parentB = g.rouletteWheelSelection(false);
		childA, childB = g.orderOneCrossOver(parentA, parentB);
		childA = childA.insertMutation(0.1); childB = childB.insertMutation(0.1);
		g.replaceWorst(Int64(g.N/2),childA,childB)
		println("k = $k \t\t\t best -> $(g.getBest())" )
		k+=1;
	end
	return g;
end

function main()
	gIni0 = GAmodel(128,512,128);
	gIni = GAmodel(128,512,128);

	for k=1:8
		gIni.data[k].x  =gIni0.data[k].x;
	end

	gOut =test1(gIni,0.0001);
	println("")
end

main()