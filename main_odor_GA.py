import os
import os.path
import numpy as np
import pickle
import copy
from subprocess import call
from read_batches import batch
from context import Parameters
from model_factory import model_factory
from loss import get_loss_per_batch

def mutate_parent(net):
    child = copy.deepcopy(net)
    for cell in child.Context:
        for key in cell.parameters:
            if np.random.rand() < P_MUT2:
                noise_t = np.random.normal(size=cell.parameters[key].shape).astype(np.float32)
                cell.parameters[key] += NOISE_STD * noise_t
    for net in child.Skill:
        for key in net.parameters:
            if np.random.rand() < P_MUT2:
                noise_t = np.random.normal(size=net.parameters[key].shape).astype(np.float32)
                net.parameters[key] += NOISE_STD * noise_t
    for key in child.DecisionMaker.parameters:
        if np.random.rand() < P_MUT2:
            noise_t = np.random.normal(size=child.DecisionMaker.parameters[key].shape).astype(np.float32)
            child.DecisionMaker.parameters[key] += NOISE_STD * noise_t
    return child


def crossover_parents(ind1, ind2):
    parent1 = copy.deepcopy(ind1)
    parent2 = copy.deepcopy(ind2)
    # For Context modules only
    for (cell1, cell2) in zip(parent1.Context, parent2.Context):
        for key in cell1.parameters:
            if np.random.rand() < P_XOVER2: # Apply Crossover
                n_rows = np.random.randint(1, cell1.parameters[key].shape[0]+1) # number of rows to exchange
                rows = np.random.choice(cell1.parameters[key].shape[0], n_rows, replace=False)
                temp = cell1.parameters[key][rows]
                cell1.parameters[key][rows] = copy.deepcopy(cell2.parameters[key][rows])
                cell2.parameters[key][rows] = copy.deepcopy(temp)
    # For Skill modules only
    for (net1, net2) in zip(parent1.Skill, parent2.Skill):
        for key in net1.parameters:
            if np.random.rand() < P_XOVER2: # Apply Crossover
                n_rows = np.random.randint(1, net1.parameters[key].shape[0]+1) # number of rows to exchange
                rows = np.random.choice(net1.parameters[key].shape[0], n_rows, replace=False)
                temp = net1.parameters[key][rows]
                net1.parameters[key][rows] = copy.deepcopy(net2.parameters[key][rows])
                net2.parameters[key][rows] = copy.deepcopy(temp)
    # For Controller only
    for key in parent1.DecisionMaker.parameters:
        if np.random.rand() < P_XOVER2: # Apply Crossover
            n_rows = np.random.randint(1, parent1.DecisionMaker.parameters[key].shape[0]+1) # number of rows to exchange
            rows = np.random.choice(parent1.DecisionMaker.parameters[key].shape[0], n_rows, replace=False)
            temp = parent1.DecisionMaker.parameters[key][rows]
            parent1.DecisionMaker.parameters[key][rows] = copy.deepcopy(parent2.DecisionMaker.parameters[key][rows])
            parent2.DecisionMaker.parameters[key][rows] = copy.deepcopy(temp)
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    return child1, child2


def binaryTourSelect(pop, fits):
    # Out of whole population, k individuals will be compared (k=3 or 5)
    arr1 = np.arange(POP_SIZE); np.random.shuffle(arr1)
    tour1 = [(arr1[i], arr1[i+1]) for i in range(0,POP_SIZE,2)]
    arr2 = np.roll(arr1, 1)
    tour2 = [(arr2[i], arr2[i+1]) for i in range(0,POP_SIZE,2)]
    tours = tour1 + tour2
    children = []
    for pair in tours:
        # compare fitnesses
        better = np.argmin([fits[pair[0]], fits[pair[1]]])
        child = copy.deepcopy( pop[pair[better]] )
        children.append( child )
    return children


###############################################################################
#----------------------------------- MAIN ------------------------------------#
###############################################################################
print(f"[DEBUG] # Available CPU cores: {os.cpu_count()}")
try:
    import torch
    print(f"[DEBUG] Torch version: {torch.__version__}")

except ModuleNotFoundError:
    print("[DEBUG] Torch not available on system.")

ctx = Parameters()


# Hyperparameters
POP_SIZE  = 20
MAX_GEN   = ctx.hp("generations", default=100)
NOISE_STD = 0.01
#STOP_GA   = 4.9
#P_MUT     = 0.5
P_MUT2    = 0.5
P_XOVER   = 0.5
P_XOVER2  = 0.5


# "baseline" -- do not reset state between trials during training
# "complete" -- reset state between trials
model_name = ctx.hp("model_name", default="complete")

from batches import features, labels, batch_ind, unique_labels

for run in range(1):
    gen = 0

    pop = [model_factory(ctx) for _ in range(POP_SIZE)]
    fits = get_loss_per_batch(ctx, pop)
    for (net, fit) in zip(pop, fits):
        net.fitVector = np.array(fit)

    while True:

        rewards = np.mean(fits, axis=1)
        reward_mean = np.mean(rewards)
        reward_min = np.min(rewards)
        best = np.argmin(rewards)
        print("%d: reward_mean=%.2f, reward_min=%.2f" % (gen, reward_mean, reward_min))

        # save temporal results
#        temp_pickle_file = "gen" + str(gen) + "_elite_CSN_Tasks1347_5pipes_run" + str(run) + ".p"
#        pickle.dump( pop[best], open( temp_pickle_file, "wb" ) )
        # if gen > 1:
        #     call("rm gen" + str(gen-1) + "*", shell=True)

#        writer.add_scalar("reward_mean", reward_mean, gen)
#        writer.add_scalar("reward_max", reward_max, gen)
        if gen >= MAX_GEN:
            print("Solved in %d steps" % gen)
            pickle_file = "elite_mCOM_Tasks12345_run" + str(run) + ".p"
            pickle.dump( pop[best], open( os.path.join(ctx.data_dir, pickle_file), "wb" ) )
            pickle_file2 = "pop_mCOM_Tasks12345_run" + str(run) + ".p"
            pickle.dump( (pop, rewards), open( os.path.join(ctx.data_dir, pickle_file2), "wb" ) )
            break
        #elif gen > MAX_GEN:
        #    print("DID NOT SOLVE!")
        #    pickle_file = "saved_elite_CSN_Tasks1347_5pipes_run" + str(run) + ".p"
        #    pickle_file2 = "saved_pop_CSN_Tasks1347_5pipes_run" + str(run) + ".p"
        #    pickle.dump(pop[best], open(pickle_file, "wb"))
        #    pickle.dump((pop, rewards), open(pickle_file2, "wb"))
        #    break

        # generate next population
        children = binaryTourSelect(pop, rewards)
        np.random.shuffle(children)
        new_pop = []
        for ind in range(0, POP_SIZE, 2):
            # VarOr (Either Crossover or Mutation)
            if np.random.random() < P_XOVER:
                child1, child2 = crossover_parents(children[ind], children[ind+1])
            else:
                child1 = mutate_parent(children[ind])
                child2 = mutate_parent(children[ind+1])
            new_pop.append(child1)
            new_pop.append(child2)

        new_fits = get_loss_per_batch(ctx, pop)
        for (net, fit) in zip(new_pop, new_fits):
            net.fitVector = np.array(fit)
        new_rewards = np.mean(new_fits, axis=1)

        # Replace the best parent with the worst child:
        best_parent = pop[best]
        worst_child = np.argmax(new_rewards)
        new_pop[worst_child] = copy.deepcopy(best_parent)
        new_fits[worst_child] = copy.deepcopy(fits[best])

        pop = copy.deepcopy(new_pop)
        fits = copy.deepcopy(new_fits)

        gen += 1

