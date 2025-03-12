import json
import os
import datetime

class Contribution:
    def __init__(self, project_name, description, date=None):
        self.project_name = project_name
        self.description = description
        self.date = date if date else datetime.datetime.now().isoformat()

    def to_dict(self):
        return {
            "project_name": self.project_name,
            "description": self.description,
            "date": self.date,
        }

    @staticmethod
    def from_dict(data):
        return Contribution(
            project_name=data["project_name"],
            description=data["description"],
            date=data["date"]
        )

class ContributionTracker:
    def __init__(self, filename='contributions.json'):
        self.filename = filename
        self.contributions = self.load_contributions()

    def load_contributions(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                data = json.load(file)
                return [Contribution.from_dict(item) for item in data]
        return []

    def save_contributions(self):
        with open(self.filename, 'w') as file:
            data = [contribution.to_dict() for contribution in self.contributions]
            json.dump(data, file, indent=4)

    def add_contribution(self, project_name, description):
        contribution = Contribution(project_name, description)
        self.contributions.append(contribution)
        self.save_contributions()

    def list_contributions(self):
        contributions_list = []
        for contribution in self.contributions:
            contributions_list.append(f"{contribution.date}: {contribution.project_name} - {contribution.description}")
        return contributions_list

    def update_contribution(self, index, project_name=None, description=None):
        if index < len(self.contributions):
            if project_name:
                self.contributions[index].project_name = project_name
            if description:
                self.contributions[index].description = description
            self.contributions[index].date = datetime.datetime.now().isoformat()
            self.save_contributions()
        else:
            raise IndexError("Contribution index out of range.")

    def remove_contribution(self, index):
        if index < len(self.contributions):
            del self.contributions[index]
            self.save_contributions()
        else:
            raise IndexError("Contribution index out of range.")

def main():
    tracker = ContributionTracker()
    
    while True:
        print("\nOpen Source Contribution Tracker")
        print("1. Add Contribution")
        print("2. List Contributions")
        print("3. Update Contribution")
        print("4. Remove Contribution")
        print("5. Exit")
        
        choice = input("Select an option: ")
        
        if choice == '1':
            project_name = input("Enter project name: ")
            description = input("Enter contribution description: ")
            tracker.add_contribution(project_name, description)
            print("Contribution added.")

        elif choice == '2':
            contributions = tracker.list_contributions()
            if contributions:
                print("Contributions:")
                for i, contribution in enumerate(contributions):
                    print(f"{i}. {contribution}")
            else:
                print("No contributions found.")

        elif choice == '3':
            index = int(input("Enter contribution index to update: "))
            project_name = input("Enter new project name (leave blank to keep current): ")
            description = input("Enter new description (leave blank to keep current): ")
            tracker.update_contribution(index, project_name if project_name else None,
                                        description if description else None)
            print("Contribution updated.")

        elif choice == '4':
            index = int(input("Enter contribution index to remove: "))
            tracker.remove_contribution(index)
            print("Contribution removed.")

        elif choice == '5':
            print("Exiting the tracker.")
            break

        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()