import matplotlib.pyplot as plt

class Plotter:
    
    def __init__(self, dir_path):
        
        self.variables = {} 
        self.dir = dir_path

    def record(self, variable_dict):

        variable_name = list(variable_dict.items())[0][0]
        variable_value = list(variable_dict.items())[0][1]

        if variable_name not in self.variables:

            self.variables[variable_name] = {}
            self.variables[variable_name]["values"] = []
            self.variables[variable_name]["x_label"] = variable_dict["x_label"]
            self.variables[variable_name]["save_freq"] = variable_dict["save_freq"]

        self.variables[variable_name]["values"].append(variable_value)
        
        if len(self.variables[variable_name]["values"]) % self.variables[variable_name]["save_freq"] == 0:
            self.save_plot(variable_name)

    def save_plot(self, var_name):

        f, ax = plt.subplots(ncols = 1, nrows = 1)

        var_values = self.variables[var_name]["values"]
        x_label = self.variables[var_name]["x_label"]
            
        n_values = len(var_values)
        x_values = list(range(n_values))

        ax.set_ylabel(var_name)
        ax.set_xlabel(x_label)
        ax.plot(x_values, var_values )

        print(f"saving plot for {var_name}")
        plt.savefig(self.dir+"/"+var_name)
        plt.close()


#plotter = Plotter()

if __name__ == "__main__":

    import random
    
    # dummy training loop that generates a loss in every timestep

    loss = 0

    for _ in range(50):
        
        # save the loss value every time record is called and save the png
        # every "save_freq" calls
        plotter.record({"training_loss":loss,
                        "x_label":"iter",
                        "save_freq": 10})

        loss += random.random()
        print(loss)


