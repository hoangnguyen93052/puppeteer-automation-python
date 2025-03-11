import random
import numpy as np
import matplotlib.pyplot as plt

class CyberWarrior:
    def __init__(self, name, attack_power, defense_power):
        self.name = name
        self.attack_power = attack_power
        self.defense_power = defense_power
        self.health = 100

    def attack(self, target):
        damage = self.attack_power - target.defense_power
        if damage > 0:
            target.health -= damage
        return damage

    def is_alive(self):
        return self.health > 0

    def __str__(self):
        return f"{self.name}: Health={self.health}, Attack={self.attack_power}, Defense={self.defense_power}"

class Simulation:
    def __init__(self):
        self.warriors = []

    def add_warrior(self, warrior):
        self.warriors.append(warrior)

    def run(self):
        while len(self.warriors) > 1:
            attacker = random.choice(self.warriors)
            target = random.choice([w for w in self.warriors if w != attacker])

            damage = attacker.attack(target)
            print(f"{attacker.name} attacks {target.name} for {damage} damage.")

            if not target.is_alive():
                print(f"{target.name} has been defeated!")
                self.warriors.remove(target)

        winner = self.warriors[0]
        print(f"The winner is {winner.name} with {winner.health} health remaining!")

def create_warriors(num_warriors):
    warriors = []
    for i in range(num_warriors):
        name = f"Warrior_{i+1}"
        attack_power = random.randint(10, 30)
        defense_power = random.randint(5, 15)
        warrior = CyberWarrior(name, attack_power, defense_power)
        warriors.append(warrior)
    return warriors

def plot_health(warriors):
    names = [w.name for w in warriors]
    health = [w.health for w in warriors]
    plt.bar(names, health, color='green')
    plt.xlabel('Warriors')
    plt.ylabel('Health')
    plt.title('Warrior Health Status')
    plt.xticks(rotation=45)
    plt.show()

def main():
    num_warriors = 5
    warriors = create_warriors(num_warriors)
    simulation = Simulation()

    for warrior in warriors:
        simulation.add_warrior(warrior)
    
    print("Starting Cyber Warfare Simulation!\n")
    simulation.run()
    plot_health(simulation.warriors)

if __name__ == "__main__":
    main()