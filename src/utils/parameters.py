class Params:
    """
    class to handle hyperparameters
    """
    
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)

    def get_params_name(self):
        return [k for k in self.__dict__]

    def gen_summary(self, separator = '#'*80):
        result = ""
        for key,value in self.__dict__.items():
            result += key + "\n" + str(value) + "\n" + separator + "\n\n"
        return result

    def save_summary(self, file_path = "summary.txt"):
        summary_str = self.gen_summary()
        with open(file_path, "w") as f:
            f.write(summary_str)
        

if __name__ == "__main__":

    params = Params(mypar1 = "oidocrop",
                    mypar2 = 69420)

    print(params.mypar1,"\n")

    summary_text = params.gen_summary()
    print("summary:")
    print(summary_text)

