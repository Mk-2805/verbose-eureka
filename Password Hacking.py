
import random
from amnesiac import blurry_memory

def selection (par , child):
  population = [par[0],par[1],par[2],par[3],child[0],child[1],child[2],child[3]]
  population = sort(population)
  #Population is sorted from worst to best passwords
  bestPop = [population[4],population[5],population[6],population[7]]
  return(bestPop)

def sort(sortlist):
  #sorting the population from worst to best
  for i in range(len(sortlist)):
      x = i + 1
      while x < len(sortlist):
          if sortlist[i][1] > sortlist [x][1]:
              temp = sortlist [x]
              sortlist [x] = sortlist[i]
              sortlist[i] = temp
              if i == 0:
                  break
              else:
                  i = i - 1
              x = i + 1
          else:
            break
  return(sortlist)

#def sort(openlist):
  #for i in range(len(openlist)):
    #cursor = openlist[i]
    #pos = i

    #while pos > 0 and openlist[pos - 1] > cursor:
        # Swap the number down the list
      #openlist[pos] = openlist[pos - 1]
      #pos = pos - 1
          # Break and do the final swap
      #openlist[pos] = cursor

      #return openlist

def crossover(par1, par2):
#creates 2 offspring using the 2 parent(half of each parent)

  #offspring = ["",""]
  #offspring[0] = offspring[0] + par1[0:5]
  #offspring[0] = offspring[0] + par2[5:10]
  #offspring[1] = offspring[1] + par2[0:5]
  #offspring[1] = offspring[1] + par1[5:10]

#creates 2 offspring by randomly selecting the characters and swapping them(crossover)
  rand = random.randint(3,5)
  offspring = ["",""]
  offspring[0] = offspring[0] + par1[0:rand]
  offspring[0] = offspring[0] + par2[rand:10]
  rand = random.randint(6,8)
  offspring[1] = offspring[1] + par2[0:rand]
  offspring[1] = offspring[1] + par1[rand:10]
  return offspring

characters = ['0','1','2','3','4','5','6','7','8','9','_','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def mutation(password,characters):
  replace = ""
  for i in range (4):
    replace = replace + characters[random.randint(0 , (len(characters) -1))]
  rand = random.randint(0,6)
  select = password[rand:rand+4]
  password = password.replace(select, replace)

  return (password)

def searching(par1,par2,par3,par4, studentid , index):
  count = 0
  parblurry = blurry_memory([par1 , par2, par3, par4 ], studentid, index)
  parating1 = parblurry[par1]
  parating2 = parblurry[par2]
  parating3 = parblurry[par3]
  parating4 = parblurry[par4]
  par = [[par1, parating1],[par2, parating2],[par3, parating3],[par4, parating4]]

  while parating4 < 0.999999:
    count = count + 1

    parblurry = blurry_memory([par[0][0] , par[1][0] , par[2][0], par[3][0] ], studentid, index)
    parating1 = parblurry[par[0][0]]
    parating2 = parblurry[par[1][0]]
    parating3 = parblurry[par[2][0]]
    parating4 = parblurry[par[3][0]]
    par = [[par[0][0], parating1],[par[1][0], parating2],[par[2][0], parating3],[par[3][0], parating4]]

    #baby1 = crossover(par[0][0],par[2][0]) #mixing the worst parent with the best one
    #baby2 = crossover(par[1][0],par[3][0]) #mixing the second worst parent with the second best one
    baby1 = crossover(par[0][0],par[1][0])
    baby2 = crossover(par[2][0],par[3][0])

    babyblur = blurry_memory([baby1[0],baby1[1],baby2[0],baby2[1]], studentid, index)
    babyrating1 = babyblur[baby1[0]]
    babyrating2 = babyblur[baby1[1]]
    babyrating3 = babyblur[baby2[0]]
    babyrating4 = babyblur[baby2[1]]
    child = [[baby1[0], babyrating1],[baby1[1], babyrating2],[baby2[0], babyrating3],[baby2[1], babyrating4]]
    par = selection(par, child)

    par[0][0] = mutation(par[0][0],characters)
    par[1][0] = mutation(par[1][0],characters)
    #print(par, count)

  #returns the best offspring(on the top of the list)
  return (par[3],count)
#example passwords
par1 = "___TEST___"
par2 = "IS_I7_L0V3"
#Passwords created manually
#par3 = "MYNAMEISMK"
#par4 = "THIS_IS_AI"
par3 = ""
par4 = ""
#half of population is made randomly
for i in range (10):
  par3 = par3 + characters[random.randint(0 , (len(characters) -1))]
  par4 = par4 + characters[random.randint(0 , (len(characters) -1))]
print("My test password is ", par1)
print("My test password is ", par2)
print("My randomly generated test password is ", par3)
print("My randomly generated test password is ", par4)
studentid = #Student ID
count = [0,0]

n = 5
for i in range (n):
  mypass0 = searching(par1 , par2 ,par3, par4, studentid , 0)
  print('My guess for the first password is ' , mypass0)
  mypass1 = searching(par1 , par2 ,par3, par4, studentid , 1)
  print('My guess for the second password is ' , mypass1)
  count[0] = count[0] + mypass0[1] + mypass1[1]
  print(i)
average = count[0]/(n*2)
print('My average out of 5 trials is:' , average)
