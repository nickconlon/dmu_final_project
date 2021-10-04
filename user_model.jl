
novice_accuracy = 0.8
expert_accuracy = 0.99
novice_certainty = 7
expert_certainty = 20
novice_availability = 1.0
expert_availability = 1.0

struct User_Model
    accuracy::Float64
    certainty::Float64
    availability::Float64
end

user_novice = User_Model(novice_accuracy, novice_certainty,novice_availability)
user_expert = User_Model(expert_accuracy, expert_certainty,expert_availability)

function sample_new_point(points,ideal,user,bad_points)
    #Determine point accuracy
    if rand(Float64,1)[1]<=user.accuracy #Accurate guess
        #Create distribution     
        choice_dist = Dirichlet(ideal*user.certainty)
        #Sample from distribution
        choice_phi = rand(choice_dist,1)
        #Find closest point
        new_point_idx,new_point_phi = find_similar_points(points,choice_phi,1,bad_points)
    else #Not accurate. Random sample
        new_point_idx = rand(range(1,length(points),step=1))
        #Make sure that randomly picked point is not the same as previously chosen
        if any(i -> string(new_point_idx) == i,bad_points)
            new_pick = true
            while new_pick # Iterate until new picked point is unique
                new_point_idx = rand(range(1,length(points),step=1))
                if ~any(i -> new_point_idx == i,bad_points)
                    new_pick = false
                end
            end
        end
        new_point_phi = [points[new_point_idx]]
        new_point_idx = [string.(new_point_idx)]
    end
    return new_point_idx,new_point_phi
end

function sample_user_response(point,ideal,user)
    #This function samples a user's response to a suggested point
    #Define distribution
    choice_dist = Dirichlet(ideal*user.certainty)
    #Sample from distribution
    choice_phi = rand(choice_dist,1)
    #Compare point similarity
    sim = similarity(point,choice_phi)

    #Generate Probabilities
    acc = user.accuracy
     # Temporary stand-in for points that are estimated to be accepted
        #p(accept) = p(accurate)p(accept|accurate) + p(not accurate)p(accept|not accurate)
        #p(deny) = p(accurate)p(deny|accurate) + p(not accurate)p(deny|not accurate)
    percentage = [acc*sim + (1-acc)*(1-sim),acc*(1-sim)+(1-acc)*sim]

    #Setup distribution
    answer_dist = SparseCat(["accept","deny"],percentage)
    return rand(answer_dist)
end
