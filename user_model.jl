
novice_accuracy = 0.8
expert_accuracy = 0.99
novice_certainty = 5
expert_certainty = 10
novice_availability = 1.0
expert_availability = 1.0

struct User_Model
    accuracy::Float64
    certainty::Float64
    availability::Float64
end

user_novice = User_Model(novice_accuracy, novice_certainty,novice_availability)
user_expert = User_Model(expert_accuracy, expert_certainty,expert_availability)

function sample_new_point(points,ideal,user)
    #Create distribution     
    choice_dist = Dirichlet(ideal*user.certainty)
    #Sample from distribution
    choice_phi = rand(choice_dist,1)
    #Find closest point
    new_point = find_similar_points(points,choice_phi,1,[])
    return new_point
end

function sample_user_response(point,ideal,user)
    #This function samples a user's response to a suggested point
    #Compare point similarity
    sim = similarity(point,ideal)

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
