
user_novice_accuracy = 0.8
user_expert_accuracy = 0.99
user_novice_availability = 1.0
user_expert_availability = 1.0

struct User_Model
    accuracy::Float64
    availability::Float64
end

user_novice = User_Model(user_novice_accuracy, user_novice_availability)
user_expert = User_Model(user_expert_accuracy, user_expert_availability)


