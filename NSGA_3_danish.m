%% Real-Time Appliances Calculations
clc; close all; clear all;

% Real-Time Loads: Light, fan, and television
n = 2; % Number of time slots in an hour (48 slots in a day)
light = zeros(1, 24*n);
fan = zeros(1, 24*n);
television = zeros(1, 24*n);

for i = 1:24*n
    if i >= 18*n
        light(i) = 0.25;
    else
        light(i) = 0.05;
    end
    if i < 8*n || i > 18*n
        fan(i) = 0.4;
    else
        fan(i) = 0.1;
    end
    if i >= 18*n + 1
        television(i) = 0.3;
    end
end
real_time_consumption = [light; fan; television]';

%% Solar PV Data Calculation
filename = 'radiation_data_june.xlsx'; % Replace with your Excel file name
hourly_radiation = xlsread(filename);
average_radiation = reshape(hourly_radiation, 24, 30);
average_radiation = mean(average_radiation');
solar_radiation_half_hourly = reshape([average_radiation; average_radiation], 1, 48);

% Constants for PV output calculation
panel_efficiency = 0.2;
panel_area = 24 * 1.5;

% Calculate solar PV output (in kW)
solar_pv_output = (1/1000) * solar_radiation_half_hourly * panel_area * panel_efficiency;

%% Base Scheduling for Shiftable Appliances (for Discomfort Calculation)
n_slots = 48; % 48 half-hour time slots
n_appliances = 10;

% Define a base schedule (preferred usage times for each appliance)
base_schedule = zeros(n_slots, n_appliances);
base_schedule(15:18, 1) = 1; base_schedule(37:40, 1) = 1; % Dishwasher
base_schedule(21:24, 2) = 1; base_schedule(39:42, 2) = 1; % Washing Machine
base_schedule(25:26, 3) = 1; base_schedule(43:44, 3) = 1; % Dryer
base_schedule(17:20, 4) = 1; base_schedule(41:44, 4) = 1; % Oven
base_schedule(17:20, 5) = 1; base_schedule(41:44, 5) = 1; % Microwave
base_schedule(11:14, 6) = 1; base_schedule(35:38, 6) = 1; % Geyser
base_schedule(19:34, 7) = 1; % Laptop
base_schedule(19:34, 8) = 1; % Desktop
base_schedule(23:24, 9) = 1; % Vacuum Cleaner
base_schedule(19:20, 10) = 1; base_schedule(39:40, 10) = 1; % Mixer

% Visualize base schedule
figure;
bar(base_schedule, 'stacked');
title('Base Schedule of Shiftable Appliances');
xlabel('Time Slots');
ylabel('Appliance Schedule');

%% Pricing and Appliance Information
pricing = [1.4, 1.4, 1.5, 1.5, 1.5, 1.5, 1.3, 1.3, 1, 1, 1.4, 1.4, 1.7, 1.7, 1.9, ...
    1.9, 2.4, 2.4, 2.4, 2.4, 2.5, 2.5, 3.7, 3.7, 3.4, 3.4, 3.3, 3.3, 4, 4, 4.7, 4.7, 4.7, 4.7, ...
    4.3, 4.3, 3.4, 3.4, 3.8, 3.8, 3.7, 3.7, 2.4, 2.4, 1.8, 1.8, 1.6, 1.6];

% Define power consumption profiles (kW) for each appliance
power_profiles = [1.2, 1, 1, 1.5, 1.3, 1.7, 0.2, 0.3, 1.2, 1.5];
power_profiles = repmat(power_profiles, n_slots, 1);

% Appliance names and operating times
appliance_names = {'Dishwasher', 'Washing Machine', 'Dryer', 'Oven', 'Microwave', ...
    'Geyser','Laptop', 'Desktop', 'Vacuum Cleaner', 'Mixer'};
operating_times = (n_slots/24) * [4, 4, 2, 4, 4, 4, 8, 8, 1, 2];

%% Battery Parameters
battery_params.capacity = 10; % Battery capacity (kWh)
battery_params.max_charge = 3; % Max charge rate (kW)
battery_params.max_discharge = -3; % Max discharge rate (kW)
battery_params.initial_soc = 5; % Initial SOC (kWh)
battery_params.charge_eff = 0.92; % Charging efficiency
battery_params.discharge_eff = 0.92; % Discharging efficiency
battery_params.delta_t = 0.5; % Time interval in hours
battery_params.operating_times = operating_times;
battery_params.n_slots = n_slots;
battery_params.n_appliances = n_appliances;

%% Run NSGA-III Optimization
nsga3_optimization(pricing, power_profiles, real_time_consumption, solar_pv_output, base_schedule, battery_params);

%% Functions
% Select the first Pareto front solutions
pareto_front = fronts{1};
num_solutions = min(12, length(pareto_front));

% Plot Pareto front (Cost vs. PAR)
figure;
scatter(fitness_values(pareto_front, 1), fitness_values(pareto_front, 2), 'filled');
xlabel('Total Cost');
ylabel('Peak-to-Average Ratio (PAR)');
title('Pareto Optimal Front');
grid on;

% Plot and display selected Pareto optimal solutions with detailed data
for i = 1:num_solutions
    figure;
    schedule = reshape(population(pareto_front(i), :), n_slots, n_appliances);
    shiftable_power = schedule .* power_profiles;

    % Calculate fitness metrics for display
    [total_cost, ~, grid_energy, pv_to_grid, battery_soc] = calculate_fitness(population(pareto_front(i), :), ...
        pricing, power_profiles, real_time_consumption, solar_pv_output);

    % Peak-to-Average Ratio (PAR) calculation for display
    par_ratio = max(grid_energy) / mean(grid_energy(grid_energy~=0));
    
    % Power consumption stacked bar plot
    total_power = [shiftable_power, real_time_consumption];
    bar(1:n_slots, total_power, 'stacked');
    hold on;
    plot(solar_pv_output, 'g-', 'LineWidth', 1.5); % PV output in green
    plot(battery_soc, 'b-', 'LineWidth', 1.5); % Battery SOC in blue
    yyaxis right;
    plot(pricing, 'k--', 'LineWidth', 1.5); % Pricing in black dashed line
    ylabel('Pricing (cents/kWh)');
    
    % Title and labels
    title(sprintf('Solution %d\nCost: %.2f, PAR: %.2f', i, total_cost, par_ratio));
    xlabel('Time Slots');
    ylabel('Power Consumption (kW)');
    colormap(parula);
    legend([appliance_names, {'Real-Time Appliances', 'PV Output', 'Battery SOC'}], 'Location', 'northeastoutside');
    hold off;
    
    % Plot grid energy import and PV export as separate figures
    figure;
    subplot(2, 1, 1);
    plot(grid_energy, 'b-', 'LineWidth', 1.5);
    title('Energy Imported from Grid');
    xlabel('Time Slots');
    ylabel('Energy (kWh)');
    
    subplot(2, 1, 2);
    plot(pv_to_grid, 'r-', 'LineWidth', 1.5);
    title('Energy Exported to Grid');
    xlabel('Time Slots');
    ylabel('Energy (kWh)');
end

% NSGA-III Optimization Function
function nsga3_optimization(pricing, power_profiles, real_time_consumption, solar_pv_output, base_schedule, battery_params)
    % Parameters
    pop_size = 5;
    max_gen = 3;

    % Initialize Population
    population = initialize_population_hybrid(pop_size, battery_params.n_slots, battery_params.n_appliances, battery_params.operating_times);
    
    % Optimization Loop
    for generation = 1:max_gen
        % Evaluate Population (Cost and PAR)
        fitness_values = evaluate_population_with_battery(population, pricing, power_profiles, ...
            real_time_consumption, solar_pv_output, battery_params, base_schedule);
        
        % NSGA-III Selection
        [fronts, rank] = non_dominated_sort_nsga3(fitness_values);
        new_population = crowded_tournament_selection_nsga3(fronts, fitness_values, pop_size, population);
        
        % Update Population for Next Generation
        population = new_population;
    end

   
end


% Crowded Tournament Selection for NSGA-III
function selected_population = crowded_tournament_selection_nsga3(fronts, fitness_values, pop_size, population)
    selected_population = [];
    front_index = 1;
    
    % Select individuals from fronts until reaching the population size
    while size(selected_population, 1) + length(fronts{front_index}) <= pop_size
        selected_population = [selected_population; population(fronts{front_index}, :)];
        front_index = front_index + 1;
        if front_index > length(fronts)
            break;
        end
    end
    
    % If there is still room in the selected population
    if size(selected_population, 1) < pop_size
        remaining = pop_size - size(selected_population, 1);
        last_front = fronts{front_index};
        
        % Calculate crowding distances for individuals in the last front
        distances = crowding_distance(fitness_values(last_front, :));
        
        % Sort the last front based on crowding distance (descending)
        [~, sorted_indices] = sort(distances, 'descend');
        
        % Select individuals based on highest crowding distance
        selected_indices = last_front(sorted_indices(1:remaining));
        selected_population = [selected_population; population(selected_indices, :)];
    end
end

% Crowding distance calculation for NSGA-III
function distances = crowding_distance(fitness_values)
    num_solutions = size(fitness_values, 1);
    num_objectives = size(fitness_values, 2);
    distances = zeros(num_solutions, 1);
    
    for i = 1:num_objectives
        [~, sorted_indices] = sort(fitness_values(:, i));
        distances(sorted_indices(1)) = inf; % Boundary solutions get maximum distance
        distances(sorted_indices(end)) = inf;
        
        min_fitness = min(fitness_values(:, i));
        max_fitness = max(fitness_values(:, i));
        range = max_fitness - min_fitness;
        
        if range == 0
            range = 1e-9; % Avoid division by zero if all values are the same
        end
        
        % Calculate normalized distances for solutions in between boundary solutions
        for j = 2:num_solutions-1
            distances(sorted_indices(j)) = distances(sorted_indices(j)) + ...
                (fitness_values(sorted_indices(j+1), i) - fitness_values(sorted_indices(j-1), i)) / range;
        end
    end
end

% Non-dominated sorting function for NSGA-III (for two objectives)
function [fronts, rank] = non_dominated_sort_nsga3(fitness_values)
    % Initialize variables
    population_size = size(fitness_values, 1);
    rank = zeros(population_size, 1);  % Store rank of each individual
    dominance_count = zeros(population_size, 1);  % Number of individuals dominating each solution
    dominated_solutions = cell(population_size, 1);  % List of solutions each individual dominates

    % Initialize first front
    fronts = {};
    fronts{1} = [];

    % Calculate dominance relationships
    for i = 1:population_size
        for j = i+1:population_size
            % Check if i dominates j
            if dominates(fitness_values(i, :), fitness_values(j, :))
                dominated_solutions{i} = [dominated_solutions{i}, j];
                dominance_count(j) = dominance_count(j) + 1;
            elseif dominates(fitness_values(j, :), fitness_values(i, :))
                % j dominates i
                dominated_solutions{j} = [dominated_solutions{j}, i];
                dominance_count(i) = dominance_count(i) + 1;
            end
        end
        % If no one dominates solution i, it is added to the first front
        if dominance_count(i) == 0
            rank(i) = 1;
            fronts{1} = [fronts{1}, i];
        end
    end

    % Form subsequent fronts
    k = 1;
    while ~isempty(fronts{k})
        next_front = [];
        for i = fronts{k}
            for j = dominated_solutions{i}
                dominance_count(j) = dominance_count(j) - 1;
                if dominance_count(j) == 0
                    rank(j) = k + 1;
                    next_front = [next_front, j];
                end
            end
        end
        k = k + 1;
        fronts{k} = next_front;
    end

    % Remove empty cells in fronts
    fronts = fronts(~cellfun('isempty', fronts));
end

% Helper function to check if solution1 dominates solution2
function is_dominant = dominates(solution1, solution2)
    % A solution dominates another if it is better in all objectives and strictly better in at least one
    is_dominant = all(solution1 <= solution2) && any(solution1 < solution2);
end

% Population Initialization with Hybrid Approach
function population = initialize_population_hybrid(pop_size, n_slots, n_appliances, operating_times)
    population = zeros(pop_size, n_slots * n_appliances);
    for i = 1:pop_size
        for j = 1:n_appliances
            chosen_slots = randperm(n_slots, operating_times(j));
            individual = zeros(n_slots, 1);
            individual(chosen_slots) = 1;
            population(i, (j-1)*n_slots + 1:j*n_slots) = individual;
        end
    end
end

% Evaluate Population with Battery Management (Cost and PAR Objectives)
function fitness_values = evaluate_population_with_battery(population, pricing, power_profiles, ...
    real_time_consumption, solar_pv_output, battery_params, base_schedule)

    fitness_values = zeros(size(population, 1), 2); % Preallocate for Cost and PAR

    for i = 1:size(population, 1)
        individual = reshape(population(i, :), battery_params.n_slots, battery_params.n_appliances);
        % Fitness Calculation
        [cost, par] = calculate_fitness_with_battery(individual, pricing, power_profiles, real_time_consumption, ...
            solar_pv_output, battery_params, base_schedule);
        
        % Ensure both cost and par are scalar values
        if isscalar(cost) && isscalar(par)
            fitness_values(i, :) = [cost, par];
        else
            error('Cost and PAR are not scalars');
        end
    end
end

% Battery-Aware Fitness Calculation
function [cost, par] = calculate_fitness_with_battery(solution, pricing, power_profiles, real_time_consumption, ...
    solar_pv_output, battery_params, base_schedule)

    % Battery Parameters
    soc = battery_params.initial_soc;
    soc_max = battery_params.capacity;
    charge_max = battery_params.max_charge;
    discharge_max = -battery_params.max_discharge;
    charge_eff = battery_params.charge_eff;
    discharge_eff = battery_params.discharge_eff;
    delta_t = battery_params.delta_t;
    avg_price = mean(pricing);
    
    % Arrays for Tracking Grid Usage
    grid_energy = zeros(battery_params.n_slots, 1);
    battery_soc = zeros(battery_params.n_slots, 1);

    % Appliance Scheduling and Total Power Consumption
    total_power = solution .* power_profiles;

    for t = 1:battery_params.n_slots
        total_demand = sum(total_power(t, :)) + sum(real_time_consumption(t, :)); % Total demand at time t
        surplus_pv = solar_pv_output(t) - total_demand; % PV surplus (if any)
        
        % Battery Charge/Discharge based on Pricing and SOC
        if surplus_pv > 0 && soc <= 0.8 * soc_max && pricing(t) <= avg_price
            % Charging Condition
            charge_power = min([surplus_pv, charge_max, (0.8 * soc_max - soc) / (charge_eff * delta_t)]);
            soc = soc + charge_power * charge_eff * delta_t;
            surplus_pv = surplus_pv - charge_power;
        elseif surplus_pv < 0 && soc >= 0.3 * soc_max && pricing(t) >= avg_price
            % Discharging Condition
            discharge_power = min([abs(surplus_pv), -discharge_max, (soc - 0.3 * soc_max) * discharge_eff / delta_t]);
            soc = soc - discharge_power / discharge_eff * delta_t;
            grid_energy(t) = -surplus_pv; % Grid energy if remaining deficit
        else
            grid_energy(t) = max(0, -surplus_pv); % Import grid energy if needed
        end
        
        % Track Battery SOC
        battery_soc(t) = soc;
    end

    % Objective Calculations (Ensure Scalars)
    cost = sum(grid_energy .* pricing(:));  % Total cost (scalar)
    if mean(grid_energy(grid_energy > 0)) > 0  % Check to prevent division by zero
        par = max(grid_energy) / mean(grid_energy(grid_energy > 0));  % PAR (scalar)
    else
        par = 0;  % Default if grid energy mean is zero
    end
    disp(['Cost size: ', num2str(size(cost))]);
    disp(['PAR size: ', num2str(size(par))]);

end


