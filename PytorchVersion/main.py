import logging
import sys

import numpy as np
import torch.cuda
import torch.nn.functional as F
from torch.autograd import Variable

import law
from Arg import Args
from Data1 import WDataset
from NetWork import Policy, Critic

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%s %p')


def cat(env, policy, arg):
	"""
    Combine the chessboard env and strategy p into a tensor with depth of 20
        Layers 1-16 are the previous environment
        The 17th layer is the reshape of p[0-360]
        The 18th layer is 1 if p.argmax==361 else 0
        Layers 19-20 are still black and white information
    @param env: batch*18*19*19
    @param policy:batch*362
    @param arg:
    @return: the combined env which size is batch*20*19*19
    """
	batch_size = arg.MINI_BATCH_SIZE
	ways = arg.ways
	cuda_str = arg.cuda_str
	batch_cat = torch.zeros(batch_size, 2, ways, ways, device=torch.device(cuda_str))
	for i in range(batch_size):
		# If the maximum v is the 361st operation, fill in all 0s, otherwise all 1s
		to_cat = torch.zeros((1, 1, ways, ways), device=torch.device(cuda_str)) if torch.argmax(
			policy[i]).item() == ways ** 2 else torch.ones((1, 1, ways, ways), device=torch.device(cuda_str))
		batch_cat[i] = torch.cat((policy[i][0:ways ** 2].reshape(1, 1, ways, ways), to_cat), dim=1)
	return torch.cat((env, batch_cat), dim=1)


def update(current_env, policy, arg, rob_list):
	"""
    Return to the  environment after executing the strategy
    @param arg:
    @param rob_list: Last take list
    @param current_env: current environment to be updated
    @param policy: strategy
    @return: env after executing the strategy
    """
	# Othello split point
	input_block_channels = arg.input_block_channels
	ways = arg.ways
	half = (input_block_channels - 2) // 2
	batch_size = arg.MINI_BATCH_SIZE
	cuda_str = arg.cuda_str
	# policy_idx is the order of p according to probability
	policy_idx = policy.cpu().detach().numpy().argsort()[::-1]
	# Try to place each batch
	new_env = current_env.clone()
	for each_batch in range(batch_size):
		batch_policy = policy_idx[each_batch]
		me, en = current_env[each_batch][0].cpu().detach().numpy(), current_env[each_batch][
			half - 1].cpu().detach().numpy()
		for each_policy in batch_policy:
			succ_flag, [me, en], rob_list = law.check([me, en], each_policy, rob_list, arg)
			if succ_flag:
				"""
                Update the board organization if the move is successful
                    new[1-8] = old[8-15]
                    new[0] = en
                    new[9-16] = old[0-7]
                    new[8] = me
                """
				new_env[each_batch][1:half] = current_env[each_batch][half:half * 2 - 1]
				new_env[each_batch][0] = torch.tensor(en.reshape(ways, ways), device=cuda_str)
				new_env[each_batch][half + 1:half * 2] = current_env[each_batch][0:half - 1]
				new_env[each_batch][half] = torch.tensor(me.reshape(ways, ways), device=cuda_str)
				break
	return new_env


def train(net, trainloader, opt_dict, arg, exp_train_need, epoch, k):
	#  TODO The update part of the model should abstract out the static method, but probably never make it :D
	cuda_str = arg.cuda_str
	batch_size = arg.MINI_BATCH_SIZE
	policy_opt, value_opt, exp_opt = opt_dict['policy_optimizer'], opt_dict['value_optimizer'], opt_dict[
		'exp_optimizer']
	rob_list = []
	actor, critic, expert = net[0].cuda(), net[1].cuda(), net[2].cuda()
	exp_total_loss, actor_total_loss, critic_total_loss = [0, 0], 0, 0
	for batch_idx, (inputs, targets, v) in enumerate(trainloader):
		inputs, targets, v = Variable(inputs).cuda(), Variable(targets).cuda(), v.cuda()
		# Experts and ac networks are trained separately
		if exp_train_need:
			if batch_idx % 3000 == 0:
				path = 'MODELS/_EXP_' + str(epoch) + str(k) + "_" + str(batch_idx)
				torch.save(expert, f=path)
			expert.train()
			exp_policy, exp_v = expert(inputs)
			length = min(arg.MINI_BATCH_SIZE, len(targets))
			# TODO
			loss_p = F.cross_entropy(exp_policy, torch.tensor(
				[torch.argmax(targets[i]).item() for i in range(length)], device=cuda_str), reduction='mean')
			loss_v = F.mse_loss(exp_v.reshape(length).to(dtype=torch.double), v)
			# logging.info("exp p for %d th batch loss: %f", batch_idx, loss_p.item())
			# logging.info("exp v for %d th batch loss: %f", batch_idx, loss_v.item())
			exp_total_loss[0] += loss_p.item()
			exp_total_loss[1] += loss_v.item()
			exp_opt.zero_grad()
			loss_p.backward(retain_graph=True)
			loss_v.backward()
			exp_opt.step()
		else:
			expert.eval()
			exp_policy, actor_policy = expert(inputs), actor(inputs)
			"""
                to_cat
                    The critic network needs to accept the previous step's strategy and chessboard to give an evaluation
                     Strategy 362 long does not meet 19*19
                     Therefore, the last no-operation is extracted into a single 19*19
            """
			if batch_idx % 3000 == 0:
				f_a, f_c = 'MODELS/actor_' + str(batch_idx), 'MODELS/critic_' + str(batch_idx)
				torch.save(actor, f=f_a)
				torch.save(critic, f=f_c)
			critic_input = cat(inputs, actor_policy[0].clone().detach(), arg)
			critic_out = critic(critic_input)
			value1 = exp_policy[1]
			new_env = update(inputs, actor_policy[0].clone().detach(), arg, rob_list)
			value2 = expert(new_env)[1]
			KL_degree = F.kl_div(F.softmax(exp_policy[0].clone().detach(), dim=1).log(),
			                     F.softmax(actor_policy[0], dim=1),
			                     reduction='batchmean')
			# v2 is the opponent's v
			"""
                KL is used to evaluate whether it is good about move at the moment. The smaller the KL, the better
                 V2 is the estimated opponent’s moving evaluation. V2 is also as small as possible. V1
                 So it can be combined as a loss
            """
			policy_error = torch.add(
										(0 - torch.sum(value1.clone().detach())) // batch_size,
			                            torch.add(
				                                    0.9 * KL_degree,
				                                    torch.sum(0.1 * value2.clone().detach()) // batch_size
			                                    )
									)
			policy_opt.zero_grad()
			policy_error.backward()
			policy_opt.step()
			logging.info(">>>p error :%f", policy_error)
			"""
                Critic is used to estimate the odds
            """
			value_loss = F.mse_loss(torch.mean(critic_out), value1.clone().detach())
			value_opt.zero_grad()
			value_loss.backward()
			value_opt.step()
			logging.info(">>> v error : %f", value_loss)
	return exp_total_loss[0] / (batch_idx + 1), actor_total_loss / (batch_idx + 1), critic_total_loss / (batch_idx + 1), \
	       exp_total_loss[1] / (batch_idx + 1)


def main(exp_train_need=False, __continue=None):
	arg = Args()
	torch.cuda.set_device(arg.gpu_id)
	assert torch.cuda.is_available(), logging.error('NO AVAILABLE GPU DEVICE')
	np.random.seed(arg.seed)
	torch.cuda.set_device(arg.gpu_id)
	torch.manual_seed(arg.seed)
	logging.info('GPU DEVICE ID = %d' % arg.gpu_id)
	logging.info('Args: %s', arg.print())
	if __continue is None:
		actor, expert, critic = Policy(arg, False), Policy(arg, True), Critic(arg)
		torch.save(expert, f='MODELS/EXP0')
	else:
		actor, _, critic = Policy(arg, False), Policy(arg, True), Critic(arg)
		expert = torch.load('MODELS/' + __continue, map_location='cuda:0')
		logging.info("MODEL HAS BEEN LOADED")
	expert.requires_grad_(exp_train_need)
	opt_dict = {
		'policy_optimizer': torch.optim.AdamW(
			actor.parameters(),
			arg.lr,
			weight_decay=arg.weight_decay,
		),
		'value_optimizer': torch.optim.AdamW(
			critic.parameters(),
			arg.lr,
			weight_decay=arg.weight_decay,
		),
		'exp_optimizer': torch.optim.AdamW(
			expert.parameters(),
			arg.lr,
			weight_decay=arg.weight_decay,
		),
	}
	logging.info("Download Data...")
	for epoch in range(arg.epochs):
		for k in range(10):
			trainloader = torch.utils.data.DataLoader(WDataset(k), batch_size=arg.MINI_BATCH_SIZE,
			                                          shuffle=True,
			                                          num_workers=4)
			logging.info("The data is loaded..")
			e_loss, a_loss, c_loss, ev_loss = train([actor, critic, expert], trainloader, opt_dict, arg, exp_train_need,
			                                        epoch, k)
			path = 'MODELS/EXP_' + str(epoch) + str(k) + "__"
			torch.save(expert, f=path)
			# logging.info(">>> e_loss : %f<<<", e_loss)
			# logging.info(">>> ev_loss : %f<<<", ev_loss)
			logging.info(">>> a_loss : %f<<<", a_loss)
			logging.info(">>> c_loss : %f<<<", c_loss)


# Program entry
if __name__ == '__main__':
	main(exp_train_need=False, __continue='EXP_16__')
