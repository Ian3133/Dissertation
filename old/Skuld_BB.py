# # Math Section for Skuld model v2
# import math
# import numpy as np

# a = 1 # the main part of what i will be changing for weights changes the size of the sigmoid without changing near zero
# b = 0  # bias will add to updating with ^ later
# m = 0.5 # the rate of change of the sigmoid 
# c = 1 # a constant I used, same as but but shift the sigmod to a place where i 0 is close to 0 (actuyll 0.04 but whatever)
# layer_bias = -0.5


# def ians_sigmoid(x, a_a, b_b, m_m, c_c, layer=0):
#     # width = max(width, 0.01)  # prevent division by 0 or instability
#     # layer_shift = layer * 0.1  # optional small bias per layer
#     # return scale * np.exp(-((x - center - layer_shift) ** 2) / (2 * width ** 2))
#     return (1/ ((1/a_a) + np.exp(((-x*m_m) + b_b + c_c + layer * layer_bias))))* a_a
#     #return 1/ ((1/a_a) + np.exp(((-x*m_m) + b_b + c_c + layer * layer_bias)))
# #print(ians_sigmoid(0.8, a, b, m, c))

# # def ians_sigmoid_new(x, a_a, b_b, m_m, c_c, layer=0):
# #     return (1/ ((1/a_a) + np.exp(((-x*m_m) + b_b + c_c + layer * layer_bias))))* a_a

# def calc_loss(output, desired):
#     # can change but not a requirement yet
#     return (0.5 * ((output - desired) ** 2))


# def d_loss_d_weight_2(n1, n2, desired):
#     # check the removal of the layer bias fo rthis one and go back to check the others as well
#     return ((n2 - desired) * (n2) * (1-(n2)) * n1) # removedall layer_bias can check layer if correct call

# def d_loss_d_weight_1(n1, n2, desired, w2, input):
#     # check the removal of the layer bias fo rthis one and go back to check the others as well
#     return ((n2 - desired) * (n2) * (1-(n2)) * n1 * w2 * (1- n1) * input)


# def d_loss_d_bias_2(n2, desired):
#     return (n2 - desired) * (n2 - layer_bias) * (1 - (n2 - layer_bias))

# def d_loss_d_bias_1(n1, n2, delta2, desired, a2):
#     #delta2 = (n2 - desired) * (n2 - layer_bias) * (1 - (n2 - layer_bias))
#     return delta2 * a2 * (n1 - layer_bias) * (1 - (n1 - layer_bias))


# def update_param(lr, grad, w):
#     w -= lr * grad
#     return w




# def forward_pass(input, a1, a2, b1, b2):
#     sig_r1 = ians_sigmoid(input, a1, b1, m, c)
#     delay_1 = sig_r1 + input
#     sig_r2 = ians_sigmoid(delay_1, a2, b2, m, c, layer=1)
#     delay_2 = sig_r2 + delay_1

#     return ([delay_1, delay_2], [sig_r1, sig_r2])


# def run_basic(input, a1, a2, b1, b2, desired, lr):   # will need to check work on bias/b and make sure everything is correct in this

#     result = forward_pass(input, a1, a2, b1, b2)
#     output = result[0][1]
#     n1 = result[0][0]
#     n2 = result[0][1]
#     print(n1, n2)
#     #loss = calc_loss(output, desired)

#     delta_a2 = - d_loss_d_weight_2(n1, n2, desired)

#     delta_a1 = - d_loss_d_weight_1(n1, n2, desired, a2, input) # trying to just input a as a the weight directly

#     new_a1 = update_param(lr, delta_a1, a1)
#     new_a2 = update_param(lr, delta_a2, a2)


#     delta_b2 = d_loss_d_bias_2(n2, desired)
#     delta_b1 = d_loss_d_bias_1(n1, n2, delta_b2, desired, b2) # trying to just input a as a the weight directly


#     new_b1 = update_param(lr, delta_b1, b1)
#     new_b2 = update_param(lr, delta_b2, b2)

#     return new_a1, new_a2, new_b1, new_b2, result


# # run_basic(0.8, 1, 1, 0, 0, 1.5, 0.1)
# class Neuron: 
#     def __init__(self, layer, number):

#         self.layer = layer
#         self.number = number
    

#         self.current_layer = 0
#         self.next_layer = 0
#         self.past_layer = 1
#         self.input_syn = 0
#         self.output_syn = 0
        
        
#         self.spikes = 0
#         #self.firetime = -1
#         self.fired = False

#         self.set_outputs(layer)

#         if self.layer == 0:
#             self.threshold = 1.0
#         else: 
#             self.threshold = 1.0  #2/3 * (self.past_layer)   #+ (self.current_layer + 1) # previous layer # of n /2 seem like a good fit  # will change to fun that reflect # of inputs and layer -- number of inputs
#         self.voltage = 0

#         #self.collected = 0
#         self.scheduled_spike = -10 # change to just runtime or somthng else
#         self.spike_time = 1000
        

#     def set_outputs(self, layer):
#         '''Called with the creation of neuron create all the correct assocations for later here'''
#         global input_layer, output_layer, syn_layer_1, syn_layer_2, hidden_layer, syn_outputs, past_layer
#         if layer == 0:
#             self.next_layer = hidden_layer
#             #self.input_syn = 0 # remains nothing
#             self.output_syn = syn_layer_1
#         elif layer == 1:
#             self.current_layer = hidden_layer
#             self.next_layer = output_layer
#             self.past_layer = input_layer
#             self.input_syn = syn_layer_1
#             self.output_syn = syn_layer_2
#         elif layer == 2:
#             self.current_layer = output_layer
#             self.next_layer = output_layer # same amount of outputs
#             self.past_layer = hidden_layer
#             self.input_syn = syn_layer_2
#             self.output_syn = syn_outputs
  

#     def send_to_syn(self, time, delay):
#         '''Sends out the times to the synapses of when they should arive at their next neuron'''
#         global l1_a, l2_a, l1_b, l2_b, m, c
#         self.fired = True
#         #print(self.layer, self.number, time, delay)
#         start = int(self.number * self.next_layer)
#         for syn in self.output_syn[start:(start+(self.next_layer))]:
#             #print('input: ', syn.input_neuron, "output ", syn.output_neuron, "on layer ", syn.layer)
#             # if self.layer == 1:
#             #     a = l1_a[syn.input_neuron][syn.output_neuron]
#             #     b = l1_b[syn.input_neuron][syn.output_neuron]
#             # if self.layer == 2:
#             #     a = l2_a[syn.input_neuron][syn.output_neuron]
#             #     b = l2_b[syn.input_neuron][syn.output_neuron]
#             syn.spike_time = ians_sigmoid(time, syn.a, syn.b, m, c, layer=syn.layer) + delay
#             #print(ians_sigmoid(time, 1, 0, m, c, layer=syn.layer) + delay)
#             #print(ians_sigmoid(time, 2, 0, m, c, layer=syn.layer) + delay)
#             #print(syn.spike_time, syn.layer)
#             #print("syn a and b", a, b)
#             #print("syna and b", syn.a, syn.b)
#             #print("syn spike time & delay ", syn.spike_time, delay)
#             #print("spike time send to syn ",syn.input_neuron,syn.output_neuron, "at time ", syn.spike_time)
#             pass
#         # for all sync going out from this one: 
#             # set call ian_sig and set that synapaces time to reach at calculated time


#     def check_arival(self):
#         '''Check first if this neuron has alreay sent or not. If it hasn't then it goes to check the synapses to see if one has arrived'''
#         if self.fired:
#             return
#         else:
#             self.check_syn_arival()


#     def check_syn_arival(self):
#         '''used in check_arival it will see if neruon is ready to act now basied on arriving spikes'''
#         global current_time
#         #print(self.layer)
#         if self.layer == 0: # only for input layer
#             if self.scheduled_spike <= current_time: # if fired already checked
#                 self.voltage += 1
#                 #print("input voltage set at time ", current_time)
#         else:

#             start = self.number * (self.current_layer -1)
#             for syn in self.input_syn[start: : self.current_layer]: # goes over all incoming synapaces
#                 if (syn.message_complete) == False: # if syn is has not yet given this neuron its message
#                     #print(self.layer, syn.layer)
#                     #print(syn.check_spike(), current_time, syn.layer)
#                     if syn.check_spike() <= current_time: # will check if current step has been reached
#                         #print(syn.check_spike(), current_time)
#                         #print(self.layer, syn.layer)

#                         #print(syn.check_spike(), current_time, self.layer)
#                         # print("layer: ", self.layer)
#                         # print("recived spike at: ",current_time)
#                         syn.message_complete = True
#                         #update neruons voltage
#                         self.voltage += 1.2

#         self.check_threshold_voltage()
        
                    
                    
#     def check_threshold_voltage(self):
#         global time_step
#         if self.voltage >= self.threshold:
#             #print(self.layer, self.voltage, self.threshold)
#             #print("self.voltage: ", self.voltage)
#             #print("threshold: ", self.threshold)
#             self.spike_time = current_time
#             #print(self.layer, self.spike_time)
#             #print("spike time: ", current_time)
#             self.send_to_syn(current_time, current_time)  # CHANGE IF LAYER NOT == 0?
#         # else:
#         #     if current_time >= (self.layer):
#         #         self.voltage += (time_step*self.past_layer) * 0.5 # will change such and whatnot just playing to start# 


# class synapse:
#     def __init__(self, input_neuron, output_neuron, layer, a, b):
#         global run_time
#         self.a = a
#         self.b = b
#         self.input_neuron = input_neuron
#         self.output_neuron = output_neuron
#         self.layer = layer
#         self.spike_time = run_time
#         self.message_complete = False
#         self.grad_a = 0
#         self.grad_b = 0

#     def set_spiketime(self, time):
#         self.spike_time = time


#     def check_spike(self):
#         return self.spike_time



# def genesis():
#     create_new_neurons_network()
#     create_synapses()
#     return


# def create_new_neurons_network():
#     count = 0
#     for i in range(len(inputs)):
#         n = Neuron(layer=0, number=i)  # will add later to add a and b and such
#         #n.set_firetime(val)
#         neurons_input.append(n)
#         n.scheduled_spike = inputs[i] 
#         #print('Neuron Input neuron ' + str(neurons_input[i].number), "scheduled time: ", n.scheduled_spike)

#     count = 0
#     for i in range (hidden_layer):
#         n = Neuron(layer=1, number=i)
#         neurons_hidden.append(n)
#         #print("Neuron at layer 1: " + str(neurons_hidden[i].number))

#     count = 0
#     for i in range (output_layer):
#         n = Neuron(layer=2, number=i)
#         neurons_output.append(n)
#         #print("Neuron at output layer: " + str(neurons_output[i].number))


# def create_synapses():
#     for input_n in neurons_input:
#         for hidden_n in neurons_hidden:
#             #print(input_n.number)
#             #print(hidden_n.number)
#             # create new object of synapics 
#             #give the object to both neurons so one can output to and the other can edit it
#             syn = synapse(input_neuron=input_n.number, output_neuron=hidden_n.number, layer=0, a= 1 + random.uniform(-0.1, 0.1), b= random.uniform(-0.1, 0.1)) 
#             syn_layer_1.append(syn)

#     for hidden_n in neurons_hidden:
#         for output_n in neurons_output:
#             #print("h", hidden_n.number)
#             #print("0" , output_n.number)
#             # create new object of synapics 
#             #give the object to both neurons so one can output to and the other can edit it
#             syn = synapse(input_neuron=hidden_n.number, output_neuron=output_n.number, layer=1, a= 1 + random.uniform(-0.1, 0.1), b= random.uniform(-0.1, 0.1)) 
#             syn_layer_2.append(syn)

    
#     for count in range(output_layer): # will check layer 
#             syn = synapse(input_neuron=count, output_neuron=count, layer=2, a=a, b=b) 
#             syn_outputs.append(syn)


#     return 0

# def run_simulation(runtime, dx):
#     global current_time, time_step
#     time_step = 1.0/dx
    
#     current_time = 0

#     for i in range(runtime*dx):
       
#         for n1 in neurons_input:
#             n1.check_arival()
#             pass 

#         for n2 in neurons_hidden:
#             n2.check_arival()
#             pass 

#         for n3 in neurons_output:
#             n3.check_arival()
#             pass

#         #print("TIME", current_time)
#         current_time += time_step


#     return 

# def reset_network():
#     global current_time
#     current_time = 0
#     for n in neurons_input: 
#         n.voltage = 0
#         n.fired = False
#         n.spike_time = run_time
#     for n in neurons_hidden:
#         n.voltage = 0
#         n.fired = False
#         n.spike_time = run_time
#     for n in neurons_output:
#         n.voltage = 0
#         n.fired = False
#         n.spike_time = run_time
#     for s in syn_layer_1:
#         s.spike_time = run_time
#         s.message_complete = False
#     for s in syn_layer_2:
#         s.spike_time = run_time
#         s.message_complete = False
