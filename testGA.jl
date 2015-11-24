#!/usr/local/bin/julia


function runga(model::GAmodel)
    reset_model(model)
    create_initial_population(model)

    while true
        evaluate_population(model)

        grouper = @task model.ga.group_entities(model.population)
        groupings = Any[]
        while !istaskdone(grouper)
            group = consume(grouper)
            group != nothing && push!(groupings, group)
        end

        if length(groupings) < 1
            break
        end

        crossover_population(model, groupings)
        mutate_population(model)
    end

    model
end

function main()
	x = generate_population
	
	
	
	
	

main()