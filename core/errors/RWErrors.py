""" Abstract errors and warnings for ECOMOD"""


class Error(Exception):
    header = ""

    def __init__(self, data):
        self.data = data
        super().__init__(self.header.format(self.data))


class Warn(Warning):
    header = ""

    def __init__(self, data):
        self.data = data
        super().__init__(self.header.format(self.data))


"""# File contains text level errors for ECOMOD"""


class NonLaTeXableError(Error):
    header = "This {file} file does not contain \\begin{document} and assumed to be not latexable.\n" \
             "Please write your model in .tex file correctly"


class NonYAMLableError(Error):
    header = "Seems like you have not formatted model in yaml notation correctly.\n" \
             "Here's error: {err}"
