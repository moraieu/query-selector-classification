import copy
import csv
import sys
import time
from datetime import datetime

ascii_offset = 64


class Case:
    def __init__(self, case_id, line, times, times2, times3, times4, datetimes):
        self.case_id = case_id
        self.line = line
        self.times = times
        self.times2 = times2
        self.times3 = times3
        self.times4 = times4
        self.datetimes = datetimes

    def trim(self, length):
        return Case(self.case_id, self.line[:length],
                    self.times[:length], self.times2[:length],
                    self.times3[:length], self.times4[:length],
                    self.datetimes[:length])

    def extract_event(self, i):
        ec = self.line[i]
        if i == len(self.line) - 1:
            t1 = 0
            t2 = 0
            t3 = 0
            t4 = 0
            dt = 0
        else:
            t1 = self.times[i]
            t2 = self.times2[i]
            t3 = self.times3[i]
            t4 = self.times4[i]
            dt = self.datetimes[i]
        return Event(ec, t1, t2,t3,t4,dt)




class Event:
    def __init__(self, event_char, time, time2, time3, time4, datetime):
        self.event_char = event_char
        self.time = time
        self.time2 = time2
        self.time3 = time3
        self.time4 = time4
        self.datetime = datetime


def load_cases(path):
    csvfile = open(path, 'r')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(spamreader, None)  # skip the headers
    # ascii_offset = 161

    lastcase = ''
    line = ''
    firstLine = True
    cases = []
    lines = []
    caseids = []
    timeseqs = []
    timeseqs2 = []
    timeseqs3 = []
    timeseqs4 = []
    datetimeseq = []
    times = []
    times2 = []
    times3 = []
    times4 = []
    datetimes = []
    numlines = 0
    casestarttime = None
    lasteventtime = None
    for row in spamreader:
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0] != lastcase:
            caseids.append(row[0])
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:
                cases.append(Case(row[0], line, times, times2, times3, times4, datetimes))
                lines.append(line)
                timeseqs.append(times)
                timeseqs2.append(times2)
                timeseqs3.append(times3)
                timeseqs4.append(times4)
                datetimeseq.append(datetimes)
            line = ''
            times = []
            times2 = []
            times3 = []
            times4 = []
            datetimes = []
            numlines += 1
        line += unichr(int(row[1]) + ascii_offset) if sys.version_info [2] == 2 else chr(int(row[1]) + ascii_offset)
        timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        timediff3 = timesincemidnight.seconds  # this leaves only time even occured after midnight
        timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()  # day of the week
        times.append(timediff)
        times2.append(timediff2)
        times3.append(timediff3)
        times4.append(timediff4)
        datetimes.append(datetime.fromtimestamp(time.mktime(t)))

        lasteventtime = t
        firstLine = False

    # add last case
    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    timeseqs4.append(times4)
    datetimeseq.append(datetimes)
    cases.append(Case(row[0], line, times, times2, times3, times4, datetimes))

    numlines += 1
    return cases


def get_training_cases(path):
    cases = load_cases(path)
    elems_per_fold = int(round(len(cases) / 3))
    cases = cases[:2 * elems_per_fold]
    return cases


def valid_cases_generator(path, maxlen):
    cases = load_cases(path)
    elems_per_fold = int(round((len(cases)) / 3))

    cases = cases[2*elems_per_fold:]

    for prefix_size in range(2, maxlen):
        print(prefix_size)
        for case in cases:
            line = case.line
            caseid = case.case_id
            times = case.times
            times2 = case.times2
            times3 = case.datetimes
            # times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            if len(times2) <= prefix_size:
                continue  # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size + maxlen])
            ground_truth_t = times2[len(times2) - 1] - times2[prefix_size - 1]
            yield caseid, prefix_size, cropped_line, cropped_times, cropped_times3, ground_truth, ground_truth_t


def get_dataset_params(cases):
    elems_per_fold = int(round(len(cases) / 3))
    fold1 = cases[:elems_per_fold]
    fold2 = cases[elems_per_fold:2 * elems_per_fold]
    cases = fold1 + fold2

    for case in cases:
        case.line = case.line + '!'
    maxlen = max(map(lambda x: len(x.line), cases))
    chars = map(lambda x: set(x.line), cases)
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    chars.remove('!')
    print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    print(indices_char)
    return elems_per_fold, maxlen, chars, target_chars, char_indices, target_char_indices, target_indices_char