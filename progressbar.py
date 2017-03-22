"""
Progress Bar code. Taken from Python Cookbook (2nd edition)
"""

import sys


class ProgressBar:
    def __init__(self, final_count, block_char='.'):
        self.final_count = final_count
        self.block_count = 0
        self.block_char = block_char
        self.output = sys.stdout

        if not self.final_count:
            return

        self.output.write('\n------------------ % Progress -------------------1\n')
        self.output.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.output.write('----0----0----0----0----0----0----0----0----0----0\n')

    def progress(self, count):
        count = min(count, self.final_count)

        if self.final_count:
            percent_complete = int(round(100.0 * count / self.final_count))
            if percent_complete < 1:
                percent_complete = 1
        else:
            percent_complete = 100

        block_count = int(percent_complete // 2)
        if block_count <= self.block_count:
            return

        for i in range(self.block_count, block_count):
            self.output.write(self.block_char)

        self.output.flush()
        self.block_count = block_count
        if percent_complete == 100:
            self.output.write("\n")


if __name__ == "__main__":
    from time import sleep

    pb = ProgressBar(8)
    for count in range(1, 9):
        pb.progress(count)
        sleep(0.2)
