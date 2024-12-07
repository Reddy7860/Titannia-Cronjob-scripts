library(quantmod) 

rm(list = ls())

setwd("~/Downloads/Reddy_Stocks_Application")
nse_data <- read.csv("~/Downloads/Reddy_Stocks_Application/data/US_30_Stocks.csv")



increment = 1

final_levels_df = data.frame("Stock" = character(0),"Rider_Bullish" = character(0),"Bullish_Level"=character(0),"Rider_Bearish"=character(0),"Bearish_Level"=character(0))


for(i in 1:nrow(nse_data)){
  
  # browser()
  stock = nse_data[i,2]
  today = Sys.Date()
  f <- function(d)if(format(d - 1, '%w') %in% c(0, 5)) Recall(d - 1) else d
  previousWorkingDay <- f(today)
  
  stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = previousWorkingDay, auto.assign = FALSE)
  # stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = "2021-01-15", auto.assign = FALSE)
  
  stock_data <- na.omit(stock_data)
  
  stock_data <- subset(stock_data, index(stock_data) < Sys.Date() + 1)
  
  stock_data <- tail(stock_data,4)
  
  print(stock_data)
  
  temp_df <- data.frame("Stock" = character(0),"Rider_Bullish" = character(0),"Bullish_Level"=character(0),"Rider_Bearish"=character(0),"Bearish_Level"=character(0))
  
  temp_df[1,'Stock'] <- stock
  
  stock_data <- data.frame(stock_data)
  
  print(stock_data)
  
  print(stock_data[4,2])
  
  if((abs((stock_data[4,2] - stock_data[3,2])/(stock_data[3,2])*100) < 0.3)){
    
    # if((abs((stock_data[4,2] - stock_data[3,2])/(stock_data[3,2])*100) < 0.5) && (abs((stock_data[3,2] - stock_data[2,2])/(stock_data[2,2])*100) < 0.5) && (abs((stock_data[2,2] - stock_data[1,2])/(stock_data[1,2])*100) < 0.5) ){
    temp_df[1,"Rider_Bullish"] = "Yes"
    # temp_df[1,"Bullish_Level"] = max(stock_data[4,2],stock_data[3,2],stock_data[2,2],stock_data[1,2])
    temp_df[1,"Bullish_Level"] = max(stock_data[4,2],stock_data[3,2])
  }
  else{
    temp_df[1,"Rider_Bullish"] = "No"
    temp_df[1,"Bullish_Level"] = 100000
  }
  if((abs((stock_data[4,3] - stock_data[3,3])/(stock_data[3,3])*100) < 0.3)){
    # if((abs((stock_data[4,3] - stock_data[3,3])/(stock_data[3,3])*100) < 0.5) && (abs((stock_data[3,3] - stock_data[2,3])/(stock_data[2,3])*100) < 0.5) && (abs((stock_data[2,3] - stock_data[1,3])/(stock_data[1,3])*100) < 0.5)){
    temp_df[1,"Rider_Bearish"] = "Yes"
    # temp_df[1,"Bearish_Level"] = min(stock_data[4,3],stock_data[3,3],stock_data[2,3],stock_data[1,3])
    temp_df[1,"Bearish_Level"] = min(stock_data[4,3],stock_data[3,3])
  }
  else{
    temp_df[1,"Rider_Bearish"] = "No"
    temp_df[1,"Bearish_Level"] = 0
  }
  # print(temp_df)
  final_levels_df = rbind(final_levels_df,temp_df)
}


write.csv(final_levels_df,paste0(getwd(),"/data/us_cowboy_data.csv"))


final_levels_df = data.frame("Stock" = character(0),"Reds_High" = numeric(0),"Reds_Low"=numeric(0))

inc = 1

for(i in 1:nrow(nse_data)){
  stock = nse_data[i,2]
  today = Sys.Date()
  f <- function(d)if(format(d - 1, '%w') %in% c(0, 5)) Recall(d - 1) else d
  previousWorkingDay <- f(today)
  
  stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = previousWorkingDay, auto.assign = FALSE)
  # stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = "2021-01-15", auto.assign = FALSE)
  
  stock_data <- na.omit(stock_data)
  
  stock_data <- subset(stock_data, index(stock_data) < Sys.Date() + 1)
  
  stock_data <- tail(stock_data,4)
  
  print(stock_data)
  
  stock_data <- as.data.frame(stock_data)
  
  rownames(stock_data) <- 1:4
  
  l1_day_range <- abs(stock_data[4,2] - stock_data[4,3])
  l2_day_range <- abs(stock_data[3,2] - stock_data[3,3])
  l3_day_range <- abs(stock_data[2,2] - stock_data[2,3])
  l4_day_range <- abs(stock_data[1,2] - stock_data[1,3])
  
  l2_day_high <- stock_data[3,2]
  l1_day_high <- stock_data[4,2]
  
  l2_day_low <- stock_data[3,3]
  l1_day_low <- stock_data[4,3]
  
  if((l1_day_range < l2_day_range) && (l1_day_range < l3_day_range) && (l1_day_range < l4_day_range)){
    if(l1_day_low > l2_day_low && l1_day_high < l2_day_high){
      
      final_levels_df[inc,"Stock"] = stock
      final_levels_df[inc,"Reds_High"] = l1_day_high
      final_levels_df[inc,"Reds_Low"] = l1_day_low
      
      
      
      inc = inc + 1
    }
    
  }
  else{
    next
  }
}

write.csv(final_levels_df,paste0(getwd(),"/data/us_reds_rocket.csv"))


final_levels_df = data.frame("Stock" = character(0),"Reds_High" = numeric(0),"Reds_Low"=numeric(0))

inc = 1

for(i in 1:nrow(nse_data)){
  stock = nse_data[i,2]
  today = Sys.Date()
  f <- function(d)if(format(d - 1, '%w') %in% c(0, 5)) Recall(d - 1) else d
  previousWorkingDay <- f(today)
  
  stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = previousWorkingDay, auto.assign = FALSE)
  # stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = "2021-01-15", auto.assign = FALSE)
  
  stock_data <- na.omit(stock_data)
  
  stock_data <- subset(stock_data, index(stock_data) < Sys.Date() + 1)
  
  stock_data <- tail(stock_data,6)
  
  # print(stock_data)
  
  stock_data <- as.data.frame(stock_data)
  
  rownames(stock_data) <- 1:6
  
  print(stock_data)
  
  l1_day_range <- abs(stock_data[6,2] - stock_data[6,3])
  l2_day_range <- abs(stock_data[5,2] - stock_data[5,3])
  l3_day_range <- abs(stock_data[4,2] - stock_data[4,3])
  l4_day_range <- abs(stock_data[3,2] - stock_data[3,3])
  l5_day_range <- abs(stock_data[2,2] - stock_data[2,3])
  l6_day_range <- abs(stock_data[1,2] - stock_data[1,3])
  
  l2_day_high <- stock_data[5,2]
  l1_day_high <- stock_data[6,2]
  
  l2_day_low <- stock_data[5,3]
  l1_day_low <- stock_data[6,3]
  
  
  if((l1_day_range < l2_day_range) && (l1_day_range < l3_day_range) && (l1_day_range < l4_day_range) && (l1_day_range < l5_day_range) && (l1_day_range < l6_day_range)){
    
    
    final_levels_df[inc,"Stock"] = stock
    final_levels_df[inc,"Reds_High"] = l1_day_high
    final_levels_df[inc,"Reds_Low"] = l1_day_low
    
    inc = inc + 1
  }
  
}

write.csv(final_levels_df,paste0(getwd(),"/data/us_reds_brahmos.csv"))


final_levels_df = data.frame("Stock" = character(0),"target" = numeric(0),"stage" = character(0))

inc = 1

for(i in 1:nrow(nse_data)){
  stock = nse_data[i,2]
  # print(stock)
  today = Sys.Date()
  f <- function(d)if(format(d - 1, '%w') %in% c(0, 5)) Recall(d - 1) else d
  previousWorkingDay <- f(today)
  
  stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = previousWorkingDay, auto.assign = FALSE)
  # stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = "2021-01-15", auto.assign = FALSE)
  
  stock_data <- na.omit(stock_data)
  
  stock_data <- subset(stock_data, index(stock_data) < Sys.Date() + 1)
  
  stock_data <- tail(stock_data,4)
  
  # print(stock_data)
  
  stock_data <- as.data.frame(stock_data)
  
  rownames(stock_data) <- 1:4
  
  l1_low <- stock_data[4,3]
  l2_low <- stock_data[3,3]
  l3_low <- stock_data[2,3]
  l4_low <- stock_data[1,3]
  
  l1_high <- stock_data[4,2]
  l2_high <- stock_data[3,2]
  l3_high <- stock_data[2,2]
  l4_high <- stock_data[1,2]
  
  if((l1_low > l2_low) && (l1_high > l2_high) && (l2_low > l3_low) && (l2_high > l3_high) && (l3_low > l4_low) && (l3_high > l4_high)){
    l1_open <- stock_data[4,1]
    l1_close <- stock_data[4,4]
    real_body <- abs(l1_open - l1_close)
    body_high <- max(l1_open,l1_close)
    if((l1_high - body_high) > 2*(real_body)){
      final_levels_df[inc,"Stock"] = stock
      final_levels_df[inc,"target"] = l1_low
      final_levels_df[inc,"stage"] = "Short"
      
      inc = inc + 1
    }
    
    
  }
  else if((l1_low < l2_low) && (l1_high < l2_high) && (l2_low < l3_low) && (l2_high < l3_high) && (l3_low < l4_low) && (l3_high < l4_high)){
    l1_open <- stock_data[4,1]
    l1_close <- stock_data[4,4]
    real_body <- abs(l1_open - l1_close)
    body_low <- min(l1_open,l1_close)
    if((l1_low - body_low) > 2*(real_body)){
      final_levels_df[inc,"Stock"] = stock
      final_levels_df[inc,"target"] = l1_high
      final_levels_df[inc,"stage"] = "Long"
      inc = inc + 1
    }
    
    
  }
  else{
    next
  }
}

write.csv(final_levels_df,paste0(getwd(),"/data/us_blackout.csv"))


final_levels_df = data.frame("Stock" = character(0),"Previous_Open" = numeric(0),"Previous_High"=numeric(0),"Previous_Low"=numeric(0),"Previous_Close"=numeric(0))

inc = 1

for(i in 1:nrow(nse_data)){
  stock = nse_data[i,2]
  today = Sys.Date()
  f <- function(d)if(format(d - 1, '%w') %in% c(0, 5)) Recall(d - 1) else d
  previousWorkingDay <- f(today)
  
  stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = previousWorkingDay, auto.assign = FALSE)
  # stock_data <- getSymbols(stock, src = "yahoo", from = "2021-01-01", to = "2021-01-15", auto.assign = FALSE)
  
  stock_data <- na.omit(stock_data)
  
  stock_data <- subset(stock_data, index(stock_data) < Sys.Date() + 1)
  
  stock_data <- tail(stock_data,1)
  
  print(stock_data)
  
  stock_data <- as.data.frame(stock_data)
  
  rownames(stock_data) <- 1
  
  final_levels_df[inc,"Stock"] = stock
  final_levels_df[inc,"Previous_Open"] = stock_data[1,1]
  final_levels_df[inc,"Previous_High"] = stock_data[1,2]
  final_levels_df[inc,"Previous_Low"] = stock_data[1,3]
  final_levels_df[inc,"Previous_Close"] = stock_data[1,4]
  
  inc = inc + 1
  
}

write.csv(final_levels_df,paste0(getwd(),"/data/us_gaps_strategy.csv"))


# final_levels_df = data.frame("Stock" = character(0),"Supply_Low" = numeric(0),"Supply_High"=numeric(0),"Supply_Time"=character(0),"Demand_Low"=numeric(0),"Demand_High"=numeric(0),"Demand_Time"=character(0))
# 
# name = "HEROMOTOCO.NS"
# 
# stock_data <- na.omit(getSymbols(name, src = "yahoo", from = "2020-10-01", to = "2021-01-22", auto.assign = FALSE))
# 
# stock_data <- data.frame(Date = index(stock_data), coredata(stock_data) )
# 
# stocks_data <- stock_data[,1:5]
# stocks_data <- cbind(newColName = rownames(stocks_data), stocks_data)
# rownames(stocks_data) <- 1:nrow(stocks_data)
# 
# stocks_data
# 
# isSupply <- function(df,i){
#   supply = (df['Open'][i-1] < df['Close'][i-1]) and (abs(df['Open'][i-1] - df['Close'][i-1]) > 0.6*(df['High'][i-1] - df['Low'][i-1])) and (abs(df['Open'][i] - df['Close'][i]) < 0.3*(df['High'][i] - df['Low'][i])) and (df['Open'][i+1] > df['Close'][i+1]) and (abs(df['Open'][i+1] - df['Close'][i+1]) > 0.6*(df['High'][i+1] - df['Low'][i+1])) 
# }

