import torch

def rebalance_portfolio(model, daily_returns, rebalance_interval):
    optimized_allocations_over_time = []
    rebalance_dates = []
    
    for i in range(rebalance_interval, len(daily_returns) + 1, rebalance_interval):
        # Ensure there's enough data to create a valid input tensor
        if i - rebalance_interval >= 0:
            input_data = daily_returns.iloc[i - rebalance_interval:i]
            current_input = torch.tensor(input_data.values, dtype=torch.float32)
            
            with torch.no_grad():
                # Use the model to predict the new allocations
                # Ensure current_input is not empty
                if current_input.size(0) > 0:
                    new_allocations = model(current_input[-1].unsqueeze(0)).numpy()
                    optimized_allocations_over_time.append(new_allocations[0])
                    
                    # Capture the rebalance date
                    rebalance_dates.append(daily_returns.index[i - 1].strftime('%Y-%m-%d'))
    
    return optimized_allocations_over_time, rebalance_dates
