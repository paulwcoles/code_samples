__author__ = "B073164"

###
# University of Edinburgh, School of Informatics
# Speech and Language Processing, December 2015
# Basic text-to-speech monophone synthesiser
# Author: Paul W. Coles
###

import os
import SimpleAudio as SA
import argparse
from nltk.corpus import cmudict
import re
import numpy as np
import sys

parser = argparse.ArgumentParser(
    description='A basic text-to-speech app that synthesises an input phrase using monophone unit selection.')
parser.add_argument('--monophones', default="monophones", help="Folder containing monophone wavs")
parser.add_argument('--play', '-p', action="store_true", default=False, help="Play the output audio")
parser.add_argument('--outfile', '-o', action="store", dest="outfile", type=str, help="Save the output audio to a file",
                    default=None)
parser.add_argument('phrase', nargs='+', help="The phrase to be synthesised")
# Arguments for extensions
parser.add_argument('--spell', '-s', action="store_true", default=False,
                    help="Spell the phrase instead of pronouncing it")
parser.add_argument('--volume', '-v', default=None, type=float,
                    help="A float between 0.0 and 1.0 representing the desired volume")

args = parser.parse_args()

class Synth(object):
    def __init__(self, wav_folder):
        self.phones = {}
        self.get_wavs(wav_folder)
        # Initialise pronunciation dictionary (always add entries for punctuation symbols)
        self.pron_dict = dict.fromkeys(['.', '?', '!'], 'double_sil')
        self.pron_dict[','] = 'sil'
        self.whole_dict = cmudict.dict()
        self.get_pron_dict(args.phrase)


    def get_wavs(self, wav_folder):
        """For each wav file in wav_folder, create entry in phones dict with filename as key.
           For each dict entry, instantiate instance of SA.Audio. Pass monophone file to load() method of that class.
        Args:
            wav_folder containing monophone recordings
        Returns:
            dict of phone filenames and audio instances
        """
        for root, dirs, files in os.walk(wav_folder, topdown=False):
            for file in files:
                if file != '.DS_Store':
                    self.phones[file] = SA.Audio()
                    self.phones[file].load('./%s/%s' % (wav_folder, file))
        return self.phones

    def get_pron_dict(self, phrase):
        """Returns a minimal pronunciation dict (contains only the pronunciations needed to synthesise the phrase)
        Args:
            Phrase to be synthesised
        Returns:
            Dictionary, word strings as keys and pronunciation as values
        """

        # Add pronunciation for spellings and numbers only if required
        for word in tokenise_phrase(phrase):
            if word not in self.pron_dict:
                if args.spell is True:
                    for char in word:
                        self.pron_dict[char] = self.whole_dict[char]
                elif re.match(r'\d+', word) is not None:
                    date_words = date_expander.months.values(), date_expander.ordinals.values(), ['the'], ['of'], ['oh']
                    number_words = number_normaliser.digits_normalised.values(),\
                            number_normaliser.tens_normalised.values(),\
                            number_normaliser.teens_normalised.values(), ['hundred'],['point'],['and'],['thousand']
                    for word_group in number_words:
                        for word in word_group:
                            self.pron_dict[word] = self.whole_dict[word][0]
                    for word_group in date_words:
                        for word in word_group:
                            self.pron_dict[word] = self.whole_dict[word][0]
        # For other tokens, add 0th pronunciation if word is in cmudict
                else:
                    try:
                        self.pron_dict[word] = self.whole_dict[word][0]
                    except:
                        print "No pronunciation available for \'%s\'. Other words synthesised." % word
        return self.pron_dict


def get_phone_seq(phrase):
    """For each token in the tokenised phrase, append its phones to a list of phones in the utterance.
    Args:
        Phrase to be synthesised
    Returns:
        phone_seq, ordered list of phones in the utterance
    """

    if args.spell:
        utt_phone_seq = spell_phrase(phrase)
    else:
        utt_phone_seq = []
        tokenised_phrase = tokenise_phrase(phrase)
        for token in tokenised_phrase:
            # Deal with date tokens
            if re.match(r'\d\d?/\d\d?/\d\d\d?\d?', token) is not None:
                for date_word in date_expander.expand_date(token):
                    for phone in S.pron_dict[date_word]:
                        phone = normalise_phone_name(phone)
                        utt_phone_seq.append(phone)
            # Deal with number tokens
            elif re.match(r'\d+', token) is not None or re.match (r'\d+\.\d+', token) is not None:
                normalised_number = number_normaliser.normalise_number(token)
                # number_normaliser function returns None and warns user where number cannot be normalised
                if normalised_number is None:
                    pass
                # else number_normaliser returns list of words in expanded number expression
                else:
                    for word in normalised_number:
                        for phone in S.pron_dict[word]:
                            phone = normalise_phone_name(phone)
                            utt_phone_seq.append(phone)
            # Append special 'phone' for punctuation tokens
            elif token in S.pron_dict:
                if S.pron_dict[token] == 'sil':
                    utt_phone_seq.append(S.pron_dict[token])
                elif S.pron_dict[token] == 'double_sil':
                    utt_phone_seq.append('sil')
                    utt_phone_seq.append('sil')
            # Remaining tokens are words
                else:
                    for phone in S.pron_dict[token]:
                        phone = normalise_phone_name(phone)
                        utt_phone_seq.append(phone)
            # Error handling for missing prons
            else:
                try:
                    if len(tokenised_phrase) == 1:
                        print "No dictionary entry exists for single word \'%s\'. No phones queued for synthesis." % token
                except:
                    print "No dictionary entry for \'%s\'. Other words synthesised." % token
    return utt_phone_seq


def tokenise_phrase(phrase):
    """Tokenise input phrase into words for dictionary look-up, licit punctuation symbols or number expressions
       Reject other input symbols
    Args:
        input phrase for synthesis
    Returns:
        ordered list of synthesis-appropriate tokens in phrase
    """
    tokenised_phrase = []
    for word in phrase:
        try:
            word = normalise_word(word)
        except:
            print 'Word \'%s\' could not be normalised. \n Input must be consist of unicode word characters, \
             question mark, exclamation mark, comma or period.' % word
        # Keep digit sequences (including dates) as is
        if re.match(r'\d+\.\d+', word) is not None:
            # Split to deal with cases of trailing punctuation (e.g. 'is Pi 3.14?')
            number_exp_as_tokens = re.split(r'(\?)|(,)|(!)', word)
            for number_exp_token in number_exp_as_tokens:
                if number_exp_token != '' and number_exp_token is not None:
                    tokenised_phrase.append(number_exp_token)
        else:
            word_as_tokens = re.split(r'(\.)|(\?)|(,)|(!)', word)
            for token in word_as_tokens:
                if token != '' and token is not None:
                    tokenised_phrase.append(token)
    return tokenised_phrase


def normalise_phone_name(phone):
    """Remove lexical stress markings and upper case from cmudict pron strings
    Args:
        cmudict standard-formatted phone name
    Returns:
        lower-case, unmarked phone name
    """
    phone = phone.lower()
    phone = re.sub(r'[0-9]?', '', phone)
    return phone


def normalise_word(word):
    """Ensures uniform word formatting before synthesis: only lower-case, unicode word and closed-set
       punctuation ('.','!','?' or ',') allowed
    Args:
        word to normalise
    Returns:
        normalised word
    """
    word = word.lower()
    if re.match(r'\w+|\.|,|\?|!', word) is not None:
        return word


def make_audio_out_array(phone_seq):
    """For each monophone in the utterance phone sequence, append to a numpy array the data for that
       monophone (from pron dict). Reformat the array using numpy's hstack() method, ready for playback.
    Args:
        phone_seq created by get_phone_seq
    Returns:
        audio_out_array (unless phone_seq is empty)
    """
    audio_out_list = []
    silence = SA.Audio()
    silence.create_noise(2500,0)
    try:
        for phone in phone_seq:
            if phone == 'sil':
                audio_out_list.append(silence.data)
            else:
                filename = str(phone)+'.wav'
                audio_out_list.append(S.phones[filename].data)
    except:
        print 'Empty phone sequence cannot be synthesised.'
        sys.exit()
    audio_out_array = np.array(audio_out_list)
    audio_out_array = np.hstack(audio_out_array)
    return audio_out_array



def spell_phrase(phrase):
    """Called when user chooses spelling not pronunciation of input.
    Args:
        phrase to synthesise
    Returns:
        phone_seq list with pronunciation for each character in each word of phrase
    """
    phone_seq = []
    for word in phrase:
        for char in word:
            try:
                for phone in S.pron_dict[char][0]:
                    phone = normalise_phone_name(phone)
                    phone_seq.append(phone)
                phone_seq.append(unicode('sil'))
            # Error handling: cmudict does not contain pronunciation for foreign characters e.g. non-Latin alphabets
            except:
                print 'Pronunciation of letter %s missing from dict' % char
    return phone_seq


class Normalised_numbers(object):
    def __init__(self):
        self.digits_normalised = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six',\
                                 '7':'seven','8':'eight','9':'nine'}
        self.teens_normalised = {'10':'ten','11':'eleven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen',\
                                 '16':'sixteen','17':'seventeen','18':'eighteen','19':'nineteen'}
        self.tens_normalised = {'2':'twenty','3':'thirty','4':'forty','5':'fifty','6':'sixty','7':'seventy',\
                                '8':'eighty','9':'ninety'}

    def normalise_number(self, number_to_normalise):
        """Called if token is number expression. Returns expansion of number expression. Handles errors of numbers
           above 3 digits before decimal point by skipping and warning user. Other functions in this class called
           by this function where appropriate.
        Args:
            number to be expanded
        Returns:
            ordered list of words in expanded number expression
        """
        self.number_halves = []
        for half in re.split(r'\.', number_to_normalise,2):
            self.number_halves.append(half)
        if len(self.number_halves[0]) <= 4:
            if len(self.number_halves) == 2:
                return self.normalise_float(self.number_halves)
            else:
                return self.normalise_integer(number_to_normalise)
        else:
            print 'Number expression %s cannot be interpreted.' % number_to_normalise
            pass


    def normalise_float(self, number_halves):
        """ Normalises the digit sequences in each half of a float number expression (digits after decimal point read
            individually, digits before decimal point expanded as integer)
        Args:
            float expression to be expanded
        Returns:
            ordered list of words in expanded float expression
        """
        normalised_float = []
        # Deal with digits before the decimal point
        if len(number_halves[0]) > 1:
            for word in self.normalise_integer(number_halves[0]):
                normalised_float.append(word)
        else:
            normalised_float.append(self.normalise_integer(number_halves[0]))
        # Add 'point' for decimal point
        normalised_float.append('point')
        # Add digits after the decimal point, truncate if too many digits
        for decimal in number_halves[1][:5]:
            for number_word in self.normalise_one_digit(decimal):
                normalised_float.append(number_word)
        if len(number_halves[1]) > 5:
            print 'Decimals in float expression (%s) truncated to five decimal places' % number_halves[1]
        return normalised_float

    def normalise_integer(self, integer):
        """ Normalises the digit sequence in an integer up to three digits in length
            Functions remaining below in this class called dependent on length of sequence for normalisation
        Args:
            integer string expression to be expanded
        Returns:
            ordered list of words in expanded integer expression
        """
        if len(integer) == 1:
            return self.normalise_one_digit(integer)
        elif len(integer) == 2:
            return self.normalise_two_digits(integer)
        elif len(integer) == 3:
            return self.normalise_three_digits(integer)
        else:
            return self.normalise_four_digits(integer)

    def normalise_one_digit(self, single_digit):
        normalised_one_digit = []
        normalised_one_digit.append(self.digits_normalised[single_digit])
        return normalised_one_digit

    def normalise_two_digits(self, two_digits):
        normalised_two_digits= []
        if two_digits[0] == '0':
            for number_word in self.normalise_one_digit(two_digits[1]):
                normalised_two_digits.append(number_word)
        elif two_digits in self.teens_normalised:
                normalised_two_digits.append(self.teens_normalised[two_digits])
        elif two_digits[1] == '0':
            normalised_two_digits.append(self.tens_normalised[two_digits[0]])
        else:
            normalised_two_digits.append(self.tens_normalised[two_digits[0]])
            for number_word in self.normalise_one_digit(two_digits[1]):
                normalised_two_digits.append(number_word)
        return normalised_two_digits

    def normalise_three_digits(self, three_digits):
        normalised_three_digits = []
        normalised_three_digits.extend((self.digits_normalised[three_digits[0]],'hundred', 'and'))
        for digit in self.normalise_two_digits(three_digits[1:]):
            if digit != 'zero':
                normalised_three_digits.append(digit)
        return normalised_three_digits

    # Four-digit normalisation necessary for year names (in Date_expander class)
    def normalise_four_digits(self, four_digits):
        normalised_four_digits = []
        normalised_four_digits.extend((self.digits_normalised[four_digits[0]], 'thousand'))
        if four_digits[1] == '0':
            normalised_four_digits.append('and')
            normalised_four_digits.extend(self.normalise_two_digits(four_digits[2:]))
        elif four_digits[1] != '0':
            for number_word in self.normalise_three_digits(four_digits[1:]):
                normalised_four_digits.append(number_word)
        elif four_digits[2] != '0':
            for number_word in self.normalise_two_digits(four_digits[2:]):
                normalised_four_digits.extend(number_word)
        elif four_digits[3] != '0':
            normalised_four_digits.append('and')
            for number_word in self.normalise_one_digit(four_digits[3]):
                normalised_four_digits.append(number_word)
        return normalised_four_digits


class Date_expander(object):
    def __init__(self):
        self.months = {'01':'january', '02':'february', '03':'march', '04':'april', '05':'may', '06':'june',\
                       '07':'july', '08':'august', '09':'september', '10':'october', '11':'november', '12':'december'}
        self.ordinals = {'30':'thirtieth', '20': 'twentieth', '10': 'tenth', '01':'first', '02':'second', '03':'third',\
                         '04':'fourth', '05':'fifth', '06':'sixth', '07':'seventh', '08':'eighth', '09':'ninth', \
                         '10':'tenth', '11':'eleventh', '12':'twelfth', '13':'thirteenth', '14':'fourteenth', \
                         '15':'fifteenth', '16':'sixteenth', '17':'seventeenth', '18':'eighteenth', '19':'nineteenth'}
        self.expanded_date = []


    def expand_date(self, date_expression):
        """Takes text date expression in format D(D) / M(M) / (YY) YY, expands to best words
        Args:
            Text date expression
        Returns:
            List of words in expression
        """
        if re.match(r'\d\d?/\d\d?/\d\d\d?\d?', date_expression) is not None:
            long_date_split = re.split(r'/', date_expression)
            # Ensure date and month numbers are two-digits
            date = self.shorthand_reformatter(long_date_split[0])
            month = self.shorthand_reformatter(long_date_split[1])
            # Check it's a plausible date
            if int(date) <= 31 and int(month) <= 12:
                self.expanded_date.append('the')
                for ordinal_word in self.make_ordinal(date):
                    self.expanded_date.append(ordinal_word)
                self.expanded_date.append('of')
                self.expanded_date.append(self.months[month])
                for year_word in self.year_expander(long_date_split[2]):
                    self.expanded_date.append(year_word)
            else:
                print 'Date expression %s could not be interpreted, was read as numbers.' % date_expression
                for date_part in long_date_split:
                    if date_part != '/':
                        for number_word in number_normaliser.normalise_number(date_part):
                            self.expanded_date.append(number_word)
        return self.expanded_date

    def shorthand_reformatter(self, shorthand):
        """For day and month numbers expressed as single digits, reformat with a leading zero
        Args:
            One-digit shorthand day/month number
        Returns:
            Same number (as string) with leading zero if appropriate
        """
        if len(shorthand) == 1:
            return '0' + shorthand
        else:
            return shorthand

    def year_expander(self, year_expression):
        """Expands year names according to British English conventions
        Args:
            Any two or four digit year expression
        Returns:
            Expanded list of words for that expression
        """
        expanded_year = []
        # for two-digit years, omit century
        if len(year_expression) == 2:
            decade = number_normaliser.normalise_number(year_expression)
            for decade_word in decade:
                if len(decade) == 1:
                    expanded_year.extend(('oh', decade_word))
                else:
                    expanded_year.append(decade_word)
        # elif between 2000 and 2009 read as normal four-digit number
        elif int(year_expression) >= 2000 and int(year_expression) <= 2009:
            for year_word in number_normaliser.normalise_number(year_expression):
                expanded_year.append(year_word)
        # else split century and decade/year, read
        else:
            for century_word in number_normaliser.normalise_number(year_expression[:2]):
                expanded_year.append(century_word)
            # first decade of every century: decade begins 'oh', ends single digit of year
            if year_expression[2] == '0':
                decade = ['oh']
                decade.extend(number_normaliser.normalise_number(year_expression[2:])[0])
            else:
                decade = number_normaliser.normalise_number(year_expression[2:])
            for decade_word in decade:
                if len(decade) == 1:
                    expanded_year.append(decade_word)
                else:
                    expanded_year.append(decade_word)
        return expanded_year

    def make_ordinal(self, cardinal_number):
        """Converts cardinal to ordinal number expression for dates
        Args:
            Any cardinal number (dictionary coverage permitting)
        Returns:
            Corresponding ordinal number as list of words
        """
        ordinal_number = []
        # Names idiosyncratic below 20, predictable above
        if int(cardinal_number) <= 20 or cardinal_number[1] == 0:
            ordinal_number.append(self.ordinals[cardinal_number])
        else:
            ordinal_number.append(number_normaliser.tens_normalised[cardinal_number[0]])
            ordinal_number.append(self.ordinals[self.shorthand_reformatter(cardinal_number[1])])
        return ordinal_number


if __name__ == "__main__":
    number_normaliser = Normalised_numbers()
    date_expander = Date_expander()
    S = Synth(wav_folder=args.monophones)
    # Initialise instance of SA.Audio as the synthesised phrase
    out = SA.Audio(rate=16000)
    # Get ordered list of phones in phrase to be synthesised
    phone_seq = get_phone_seq(args.phrase)
    # Assign final phone sequence array to data variable of the 'out' instance
    out.data = make_audio_out_array(phone_seq)
    # Handle user preferences from command line args
    if args.volume is not None:
        out.rescale(args.volume)
    elif args.play:
        out.play()
    elif args.outfile is not None:
        with open(args.outfile, 'w') as outfile:
            out.save(outfile)
    # Check user has either selected to play OR write outfile
    else:
        print 'Pass at least one argument of \'-p\' or \'-o\' at command line.'
        print parser.format_usage()
        parser.exit()
