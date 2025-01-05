"""impsy.interaction_config: Functions for using imps as an interactive music system. This version uses a config file instead of a CLI."""

import logging
import time
import datetime
import numpy as np
import queue
from threading import Thread
import click
from .utils import mdrnn_config, get_config_data, print_io
import impsy.impsio as impsio
from pathlib import Path
from impsy.structure import PredictionTree
import impsy.heuristics as heuristics

np.set_printoptions(precision=2)

INTERACTION_MODES = {
    "callresponse": {
        "user_to_rnn": True,
        "rnn_to_rnn": False,
        "rnn_to_sound": False,
    },
    "polyphony": {
        "user_to_rnn": True,
        "rnn_to_rnn": False,
        "rnn_to_sound": True,
    },
    "battle": {
        "user_to_rnn": False,
        "rnn_to_rnn": True,
        "rnn_to_sound": True,
    },
    "useronly": {
        "user_to_rnn": False,
        "rnn_to_rnn": False,
        "rnn_to_sound": False,
    },
}


def setup_logging(dimension: int, location="logs", delay_file_open=True):
    """Setup a log file and logging, requires a dimension parameter"""
    log_date = datetime.datetime.now().isoformat().replace(":", "-")[:19]
    log_name = f"{log_date}-{dimension}d-mdrnn.log"
    log_file = Path(location) / log_name
    # make sure logging directory exists.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_format = "%(message)s"
    
    # Set up a specific logger for IMPSY
    file_handler = logging.FileHandler(log_file, delay=delay_file_open)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger("impsylogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    click.secho(f"Logging enabled: {log_name}", fg="green")
    return logger


def log_interaction(source: str, values: np.ndarray, logger: logging.Logger):
    value_string = ",".join(map(str, values))
    logger.info(f"{datetime.datetime.now().isoformat()},{source},{value_string}")


def close_log(logger: logging.Logger):
    for handler in logger.handlers:
        logger.removeHandler(handler)
        handler.close()


def build_network(config: dict):
    """Build the MDRNN, uses a high-level size parameter and dimension."""
    from . import mdrnn

    try:
        dimension = config["model"]["dimension"]
    except Exception as e:
        click.secho(f"MDRNN: Couldn't find a dimension in your config. Please add one!", fg="red")
        raise

    try:
        model_size = config['model']['size']
        click.secho(f"MDRNN: Using {model_size} model.", fg="green")
    except Exception as e:
        model_size = "s"
        click.secho(f"MDRNN: Couldn't find a model size in your config, using {model_size}.", fg="red")

    model_config = mdrnn_config(model_size)
    units = model_config["units"]
    mixtures = model_config["mixes"]
    layers = model_config["layers"]

    try:
        model_file = Path(config["model"]["file"])
    except Exception as e:
        click.secho(f"MDRNN: Couldn't find a model file in your config. Loading dummy model.", fg="red")
        model_file = Path(".")
    
    if model_file.suffix == ".keras" or model_file.suffix == ".h5":
        click.secho(f"MDRNN Loading from .keras or .h5 file: {model_file}", fg="green")
        model = mdrnn.KerasMDRNN(model_file, dimension, units, mixtures, layers)
    elif model_file.suffix == ".tflite":
        click.secho(f"MDRNN Loading from .tflite file: {model_file}", fg="green")
        model = mdrnn.TfliteMDRNN(model_file, dimension, units, mixtures, layers)
    else:
        click.secho(f"MDRNN Loading dummy model: {model_file}", fg="yellow")
        model = mdrnn.DummyMDRNN(model_file, dimension, units, mixtures, layers)

    model.pi_temp = config["model"]["pitemp"]
    model.sigma_temp = config["model"]["sigmatemp"]
    click.secho(f"MDRNN Loaded.", fg="green")
    return model


class InteractionServer(object):
    """Interaction server class. Contains state and functions for the interaction loop."""

    def __init__(self, config: dict, log_location: str = "logs"):
        """Initialises the interaction server including loading the config from a config.toml file."""
        click.secho("Preparing IMPSY interaction server...", fg="yellow")
        self.config = config

        ## Load global variables from the config file.
        self.verbose = self.config["verbose"]
        self.dimension = self.config["model"][
            "dimension"
        ]  # retrieve dimension from the config file.
        self.mode = self.config["interaction"]["mode"]

        ## Set up log
        self.log_location = log_location
        self.logger = setup_logging(self.dimension, location=self.log_location)

        ## Set up IO.
        self.senders = []

        if "midi" in self.config:
            midi_sender = impsio.MIDIServer(
                self.config, self.construct_input_list, self.dense_callback
            )
            self.senders.append(midi_sender)
        
        if "websocket" in self.config:
            websocket_sender = impsio.WebSocketServer(
                self.config, self.construct_input_list, self.dense_callback
            )
            self.senders.append(websocket_sender)
        
        if "osc" in self.config:
            osc_sender = impsio.OSCServer(
                self.config, self.construct_input_list, self.dense_callback
            )
            self.senders.append(osc_sender)
        
        if "serial" in self.config:
            self.senders.append(impsio.SerialServer(self.config, self.construct_input_list, self.dense_callback))

        if "serialmidi" in self.config:
            self.senders.append(impsio.SerialMIDIServer(self.config, self.construct_input_list, self.dense_callback))

        # connect all the senders
        for sender in self.senders:
            sender.connect()

        # Import MDRNN
        click.secho("Importing MDRNN.", fg="yellow")
        start_import = time.time()
        import impsy.mdrnn as mdrnn

        click.secho(
            f"Done in {round(time.time() - start_import, 2)}s.",
            fg="yellow",
        )

        # Interaction Loop Mapping
        if self.mode in INTERACTION_MODES:
            mode_mapping = INTERACTION_MODES[self.mode]
        else:
            click.secho(f"Warning: could not set {self.mode} mode, using default.", fg="yellow")
            mode_mapping = INTERACTION_MODES["useronly"]
        click.secho(f"Config: {self.mode} mode.", fg="blue")
        self.user_to_rnn = mode_mapping["user_to_rnn"]
        self.rnn_to_rnn = mode_mapping["rnn_to_rnn"]
        self.rnn_to_sound = mode_mapping["rnn_to_sound"]

        # Set up runtime variables.
        self.interface_input_queue = queue.Queue()
        self.rnn_prediction_queue = queue.Queue()
        self.rnn_output_buffer = queue.Queue()
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = mdrnn.random_sample(out_dim=self.dimension)
        self.rnn_prediction_queue.put_nowait(
            mdrnn.random_sample(out_dim=self.dimension)
        )
        self.call_response_mode = "call"

        # Set up structural variables
        self.rnn_output_memory_size = 15
        self.rnn_output_memory = []
        self.rnn_prediction_tree = None



    def send_back_values(self, output_values):
        """sends back sound commands to the MIDI/OSC/WebSockets outputs"""
        output = np.minimum(np.maximum(output_values, 0), 1)
        if self.verbose:
            print_io("out", output, "green")
        for sender in self.senders:
            sender.send(output)

    def dense_callback(self, values) -> None:
        """insert a dense input list into the interaction stream (e.g., when receiving OSC)."""
        values_arr = np.array(values)
        if self.verbose:
            print_io("in", values_arr, "yellow")
        log_interaction("interface", values_arr, self.logger)
        dt = time.time() - self.last_user_interaction_time
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = np.array([dt, *values_arr])
        assert (
            len(self.last_user_interaction_data) == self.dimension
        ), "Input is incorrect dimension. set dimension to %r" % len(
            self.last_user_interaction_data
        )
        # These values are accessed by the RNN in the interaction loop function.
        self.interface_input_queue.put_nowait(self.last_user_interaction_data)

    # Todo this is the "callback" for our IO functions.
    def construct_input_list(self, index: int, value: float) -> None:
        """constructs a dense input list from a sparse format (e.g., when receiving MIDI)"""
        # set up dense interaction list
        int_input = self.last_user_interaction_data[1:]
        int_input[index] = value
        new_input = int_input[:1] #TODO REMOVE
        print("VALUES:", new_input) #TODO REMOVE
        # log
        if self.verbose:
            click.secho(f"in: {values}", fg="yellow")
        logging.info(
            "{1},interface,{0}".format(
                ",".join(map(str, new_input)), datetime.datetime.now().isoformat() #TODO change back to int_input
            )
        )
        # put it in the queue
        dt = time.time() - self.last_user_interaction_time
        self.last_user_interaction_time = time.time()
        self.last_user_interaction_data = np.array([dt, *values])
        assert (
            len(self.last_user_interaction_data) == self.dimension
        ), "Input is incorrect dimension, set dimension to %r" % len(
            self.last_user_interaction_data
        )
        # These values are accessed by the RNN in the interaction loop function.
        self.interface_input_queue.put_nowait(self.last_user_interaction_data)
        # Send values to output if in config
        if self.config["interaction"]["input_thru"]:
            # This is where outputs are sent via impsio objects.
            output_values = np.minimum(
                np.maximum(self.last_user_interaction_data[1:], 0), 1
            )
            self.send_back_values(output_values)

    def make_prediction(self, neural_net):
        """Part of the interaction loop: reads input, makes predictions, outputs results"""
        # First deal with user --> MDRNN prediction
        if self.user_to_rnn and not self.interface_input_queue.empty():
            print("User to RNN")
            item = self.interface_input_queue.get(block=True, timeout=None)
            print("Item:", item)
            rnn_output = neural_net.generate_touch(item)
            print("RNN Output:", rnn_output)
            if self.rnn_to_sound:
                self.rnn_output_buffer.put_nowait(rnn_output)
            self.interface_input_queue.task_done()

        # Now deal with MDRNN --> MDRNN prediction.
        if (
            self.rnn_to_rnn
            and self.rnn_output_buffer.empty()
            and not self.rnn_prediction_queue.empty()
        ):
            #print("RNN to RNN")
            item = self.rnn_prediction_queue.get(block=True, timeout=None)
            #print("Item:", item)
            # If memory is length zero, set memory to the current item.
            if not self.rnn_output_memory:
                self.rnn_output_memory = [item]
            # # start timer
            # start_time = time.time()
            # self.rnn_prediction_tree.build_tree(self.rnn_output_memory, neural_net.generate_touch)
            # # End timer
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time}")
# 
            # # Print the sampled branches
            # self.rnn_prediction_tree.rank_branches(heuristics.four_note_repetition)
            # #for i, (branch, heuristic_value) in enumerate(sampled_branches):
            # #    print(f"Branch {i + 1} Heuristic Value: {heuristic_value:.4f}):")
            # #    print(branch)
            # #    print(f"Branch length: {len(branch)}")
            # #    print()
            # # Get the branch with the highest heuristic value
            # best_branch = self.rnn_prediction_tree.best_branch
            # #print("Best Branch:", best_branch)
            # # Get the item from memory length + 1 from the best branch
            # rnn_output = best_branch[0][len(self.rnn_output_memory)]
            # #lstm_states = best_branch[0][len(self.rnn_output_memory)][1]
            
            start_time = time.time()
            for i in range(1000): #TODO: Tested as 0.15ms on average, remove
                rnn_output = neural_net.generate_touch(item)
            end_time = time.time()
            print("RNN Output:", rnn_output)
            print(f"Time taken: {end_time - start_time}")

            self.rnn_output_memory.append(rnn_output)
            if len(self.rnn_output_memory) > self.rnn_output_memory_size:
                self.rnn_output_memory.pop(0)
            #print("Memory:", self.rnn_output_memory)
            self.rnn_output_buffer.put_nowait(
                rnn_output
            )  # put it in the playback queue.
            self.rnn_prediction_queue.task_done()

    def monitor_user_action(self):
        """Handles changing responsibility in Call-Response mode."""
        # Check when the last user interaction was
        dt = time.time() - self.last_user_interaction_time
        if dt > self.config["interaction"]["threshold"]:
            # switch to response modes.
            self.user_to_rnn = False
            self.rnn_to_rnn = True
            self.rnn_to_sound = True
            if self.call_response_mode == "call":
                click.secho("switching to response.", bg="red", fg="black")
                self.call_response_mode = "response"
                while not self.rnn_prediction_queue.empty():
                    # Make sure there's no inputs waiting to be predicted.
                    self.rnn_prediction_queue.get()
                    self.rnn_prediction_queue.task_done()
                self.rnn_prediction_queue.put_nowait(
                    self.last_user_interaction_data
                )  # prime the RNN queue
        else:
            # switch to call mode.
            self.user_to_rnn = True
            self.rnn_to_rnn = False
            self.rnn_to_sound = False
            if self.call_response_mode == "response":
                click.secho("switching to call.", bg="blue", fg="black")
                self.call_response_mode = "call"
                # Empty the RNN queues.
                while not self.rnn_output_buffer.empty():
                    # Make sure there's no actions waiting to be synthesised.
                    self.rnn_output_buffer.get()
                    self.rnn_output_buffer.task_done()
                # send MIDI noteoff messages to stop previous sounds
                # TODO: this could be framed as "control switching"

    def playback_rnn_loop(self):
        """Plays back RNN notes from its buffer queue. This loop blocks and should run in a separate thread."""
        while True:
            item = self.rnn_output_buffer.get(
                block=True, timeout=None
            )  # Blocks until next item is available.
            dt = item[0]
            # click.secho(f"Raw dt: {dt}", fg="blue")
            x_pred = np.minimum(np.maximum(item[1:], 0), 1)
            dt = max(dt, 0.001)  # stop accidental minus and zero dt.
            dt = dt * self.config["model"]["timescale"]  # timescale modification!
            # click.secho(f"Sleeping for dt: {dt}", fg="blue")

            time.sleep(dt)  # wait until time to play the sound
            # put last played in queue for prediction.
            self.rnn_prediction_queue.put_nowait(
                np.concatenate([np.array([dt]), x_pred])
            )
            if self.rnn_to_sound:
                # Send predictions to outputs via impsio objects
                self.send_back_values(x_pred)
                if self.config["log_predictions"]:
                    log_interaction("rnn", x_pred, self.logger)
            self.rnn_output_buffer.task_done()


    def shutdown(self):
        """Close IO and logs and prepare to exit."""
        for sender in self.senders:
            sender.disconnect()
        close_log(self.logger)


    def serve_forever(self):
        """Run the interaction server opening required IO."""
        click.secho("Preparing MDRNN.", fg="yellow")
        net = build_network(self.config)
        if self.config["model"]["file"] != "":
            net.load_model(
                model_file=self.config["model"]["file"]
            )  # load custom model.
        else:
            net.load_model()  # try loading from default file location.

        # Threads
        click.secho("Preparing MDRNN thread.", fg="yellow")
        rnn_thread = Thread(
            target=self.playback_rnn_loop, name="rnn_player_thread", daemon=True
        )

        # Start threads and run IO loop
        try:
            rnn_thread.start()
            click.secho("RNN Thread Started", fg="green")
            while True:
                self.make_prediction(net)
                if self.config["interaction"]["mode"] == "callresponse":
                    for sender in self.senders:
                        sender.handle()  # handle incoming inputs
                    self.monitor_user_action()
        except KeyboardInterrupt:
            click.secho("\nCtrl-C received... exiting.", fg="red")
            rnn_thread.join(timeout=1.0)
            self.shutdown()
        finally:
            click.secho("\nIMPSY has shut down. Bye!", fg="red")


@click.command(name="run")
@click.option('--config', '-c', default='config.toml', help='Path to a .toml configuration file.')
@click.option('--logdir', '-l', default='logs', help='Path to a directory for logs.')
def run(config: str, logdir: str):
    """Run IMPSY interaction system with MIDI, WebSockets, and OSC."""
    click.secho("IMPSY Starting up...", fg="blue")
    config_data = get_config_data(config)
    # TODO: have some way set log dir in config as well?
    interaction_server = InteractionServer(config_data, log_location=logdir)
    interaction_server.serve_forever()
