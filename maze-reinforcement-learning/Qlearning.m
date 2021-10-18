%% Initialization
%  Initialize the world, Q-table, and hyperparameters
world_index = 3;
world = gwinit(world_index);
gwdraw();

% actions: down, up, right, left
Q = rand(world.ysize,world.xsize,4);

iterations = 5000;
learning_rate = 0.3;
discount_factor = 0.9;

%% Training loop
%  Train the agent using the Q-learning algorithm.

for i = 1 : iterations
    disp("iteration " + i)
    world = gwinit(world_index);
    state = gwstate();
    while ~state.isterminal
        y_pos = state.pos(1);
        x_pos = state.pos(2);

        action = chooseaction(Q, y_pos, x_pos, [1 2 3 4], [0.25 0.25 0.25 0.25], getepsilon(i, iterations));
        state = gwaction(action);
        
        % if we're in the first three worlds and action is invalid (e.g. going into wall), set Q-value to -infinity
        if((world_index <= 3) && (~state.isvalid))
            new_q_val = -inf;
        else
            y_new = state.pos(1); 
            x_new = state.pos(2);
            new_q_val = (1-learning_rate)*Q(y_pos, x_pos, action) + learning_rate * (state.feedback + discount_factor * max(Q(y_new,x_new,:)));
        end
        Q(y_pos, x_pos, action) = new_q_val;
    end
    % Q-value should be 0 for all actions at the goal
    Q(state.pos(1), state.pos(2),:) = 0;
end

%% Plot the optimal policy and V-values after finished training
P = getpolicy(Q);
figure();
gwdraw();
gwdraw("Policy", P, "Episode", iterations);
figure();
imagesc(getvalue(Q));
colorbar;

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0; always pick
%  the optimal action.

gwinit(world_index);
figure();
gwdraw();
state = gwstate();
while ~state.isterminal
    y_pos = state.pos(1);
    x_pos = state.pos(2);
    action = chooseaction(Q, y_pos, x_pos, [1 2 3 4], [0.25 0.25 0.25 0.25], 0);
    state = gwaction(action);
    gwdraw();
end
