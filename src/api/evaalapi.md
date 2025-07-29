# IPIN competition interface

A user is given one or more unique trial names `TRIAL` and a server URL for accessing the web API.

Accessing the server URL with a browser shows pointers to documentation and source code.

This API is used for `offsite` trials, in contrast with `onsite` trials, which measure the performance of physical devices.

In most instances, the following documentation distinguishes among `online` and `offline` trials.  Online trials emulate an onsite real-time system trial, where a physical device continuously reads sensors data and periodically produces position estimates.  Offline trials emulate a batch application which reads a whole log containing multiple sensors data collected during a time interval and computes estimates for the whole trial at once.


## API overview for online trials

For each trial, the server maintains a `trial timestamp` and a position estimate. Initially, the `trial timestamp` is set to the timestamp of the first data line.  The trial has not started, and the client can optionally verify it with

    GET /TRIAL/state

which returns a string starting with `0,-1`, meaning that the trial has not started yet.  To start the trial, the client iterates through a sequence of sending position estimates and getting data until the end of the trial.

In the usual workflow, the client repeatedly sends a new position estimate every 0.5&nbsp;s (the recommended value) and gets sensors data by repeatedly issuing the `nexdata` command (that is, HTTP request) like in this example

    GET /TRIAL/nextdata?position=10.422057,43.718278,1&horizon=0.5

For each `nexdata` command the server sets the `position` estimate relative to the `trial timestamp`, then advances the `trial timestamp` by the `horizon` (0.5&nbsp;s in the example), then returns a number of data lines whose timestamps are non-decreasing.  The data timestamps belong to the interval from the previous `trial timestamp` (included) to the current `trial timestamp` (excluded).

If the client receives no data lines, one of the following apply:

 - no data are available for the requested interval, but data are available further on; the server sets the position estimate; updates the `trial timestamp`; returns code 200
 - no more data are available for this trial because the trial has finished normally; the server ignores the position estimate; does not update the `trial timestamp`; returns code 405
 - the trial has finished because of a timeout; the server ignores the position estimate; does not update the `trial timestamp`; returns code 405


### Timeout in online trials

Ideally, data between client and server should be exchanged in real time, as it is in onsite real-time competitions, but this is not generally possible because of network processing delays and network latency.  As a compromise, a timeout is defined as follows.  Times are in seconds.

 - Let `p` be the clock time of the previous `nextdata` command
 - Let `h` be the `horizon` of the previous `nextdata` command
 - Let `c` be the clock time of the current `nextdata` command

For a physical real-time system we always have

    h == c-p

To account for processing delays and network latency, we define

 - `V` as the trial time slowdown factor
 - `S` as the slack time
 - `s` as the remaining slack, which is initially set to `S`

Whenever it receives a `nextdata` command, the server computes

    s += V*h - (c-p)
    if s>S then s=S
    if s<0 then Timeout

In any moment when `s>0` the time remaining until timeout is

    rem = p + V*h + s - current_clock

The server does not serve scoring (non-reloadable) trials faster than real time if V > 2.


## API overview for offline trials

Initially, the trial has not started.  The client can optionally verify it with

    GET /TRIAL/state

which returns a string starting with `0,-2`, meaning that the trial has not started yet.

As the first step, the client gets the sensors data all at once by issuing the command (that is, HTTP request)

    GET /TRIAL/nextdata?offline

to which the server returns a number of data lines whose timestamps are in non-decreasing order.

As the second and last step, the client posts the estimated positions all at once with the command

    POST /TRIAL/estimates

followed by the estimates as ASCII lines, one per line.  Each estimate is a timestamped position.

The second step must be executed no later than `S` seconds after the first one to avoid a timeout.


## Commands

This web API does not follow REST principles: most commands are HTTP requests using `GET` which are not idempotent; specifically, both the `nextdata` and `reload` commands can explicitly change the state of the system.  This architectural choice is done to reflect the normal workflow and minimise the number of web calls.  Moreover, the system state depends on the number of milliseconds since the first `nextdata` command, so generally speaking the system state changes even when no command is given.


### GET /TRIAL/state

Returns an ASCII one-line string of 7 comma-separated numbers followed by an unquoted string.

#### Online trials
The 'V', 'S' and 'REM' values are relative to timeout computation.

 - `0,-1,V,S,0,0,0,POS`
   the trial has not started; `POS` is the initial position (a string)
 - `TS,REM,V,S,p,h,PTS,POS` (numbers are floats, times are in seconds, `POS` is a string)
   the trial is running; `TS` is the `trial timestamp`, `PTS` is the timestamp relative to the last position estimate `POS`; all numbers are non-negative apart from `REM`, which is negative if `nextdata` would return timeout
 - `-1,REM,V,S,p,h,PTS,POS`
   the trial has finished; `REM` is the slack time at the time of the last `nextdata` command: it is not less than 0 if the trial finished normally and less than 0 if it finished by timeout; the other parameters are those in effect after the last `nextdata` command issued when the trial was running

#### Offline trials
 - `0,-2,0,S,0,0,0,POS`
   the offline trial has not started; `S` is the max interval between a `nextdata` and an `estimates` POST command; `POS` is the initial position (a string)
 - `TS,REM,0,S,p,-2,0,POS` (numbers are floats, times are in seconds, `POS` is a string)
   the trial is running; `TS` is the `trial timestamp`; all numbers are non-negative apart from -2 in the sixth position and `REM`, which is negative if an `estimates` POST command would return timeout
 - `-1,REM,0,S,p,-2,PTS,POS`
   the trial has finished; `REM` is the remaining time at the time of the `estimates` POST command: it is not less than 0 if the trial finished normally and less than 0 if it finished by timeout; the other parameters are those in effect after the last `estimates` POST command

#### All trials

In Python, the string can be read with type checking by using
`parse("{trialts:f},{rem:f},{V:f},{S:f},{p:f},{h:f},{pts:f},{pos:S}", string)` from the `parse` library.

Returns code 404 if `TRIAL` does not exist.


### Data format

The `nexdata` command for both online and offline trials returns an ASCII multiline string.  The data format is trial-dependent with the following contraints:

 - data is served with `Content-type` set to `text/csv`
 - the separator is a trial-dependent character
 - the first numeric field is the line's timestamp
 - each line's timestamp is not less than the previous line's


### GET /TRIAL/nextdata (position, horizon)

Used for online trials.  This command must be called repeatedly until end of trial.  Parameters `position` and `horizon` are both allowed and both optional.

Sets the trial position estimate, advances the `trial timestamp`.  Upon the first `nextdata` command, the `trial timestamp` is set to the `initial timestamp`, which is the timestamp of the first data line.

With parameter `position=POS` a position estimate is set relative to the `trial timestamp`.  `POS` is a string in a trial-dependent format.  For example, it may be made of three comma-separated numbers `longitude,latitude,height`.  Longitude and latitude are in degrees, ranging from -180.0 to 180.0.  Height is trial-dependent, for example it can be defined as an integer number indicating the floor number, where 0 is the ground floor, positive numbers are floors above ground and negative numbers are floors below ground.  The trial position estimate is set to `POS` if the trial is running and the `trial timestamp` is greater than the `initial timestamp`: the position estimate at the `initial timestamp` is defined as the initial position.

After possibly setting the position estimate, data lines for the next `H` seconds are returned and the `trial timestamp` is updated.  Parameter `horizon=H` sets the horizon to a non-negative float (default `horizon=0.5`).  For example when no `horizon` parameter is present, the default of 0.5&nbsp;s is assumed, and all data lines are returned with timestamps greater or equal to `trial timestamp` and smaller than `trial timestamp` plus half a second.  Finally, the `trial timestamp` advances by `H` seconds.

Hint: to get the `initial timestamp` of a trial in not-started state without advancing the `trial timestamp` you can issue an initial `nextdata` command with `horizon=0`, thus starting the trial, followed by a `state` command.

Returns code 405 if the trial has finished, whether normally or because of a timeout; the position estimate is ignored; the `trial timestamp` does not change; returns the state data as with command `state`.

Returns code 422 if any parameter does not conform to the above rules; the position estimate is ignored; the `trial timestamp` does not change; no data are returned.

Returns code 423 if non-reloadable, V>2 and the client is faster than real time; the position estimate is ignored; the `trial timestamp` does not change; no data are returned.


### GET /TRIAL/nextdata (offline)

Used for offline trial.  This command must be called only once at start of trial.  Parameter `offline` is compulsory; it does not require a value.

All data lines are returned with a single call.

Returns code 405 when called more than once; returns the state data as with command `state`.

Returns code 422 if this is not an offline trial.


### GET /TRIAL/reload (keeplog)

This command is used to set the state of the trial to not started.  It only works for testing (reloadable) trials.  It does not work for scoring (non-reloadable) trials.

If the trial is reloadable, put it in the not started state and return its state.  Same if the trial
is non-reloadable and no log was generated.

Delete the log file unless using parameter `keeplog`.

Returns code 422 if this is a scoring (non-reloadable) trial.


### GET /TRIAL/estimates

Returns the estimates in csv format with header.

The first line is the string: `pts,c,h,s,pos`.  The following lines represent each a position estimate containing 5 comma-separated fields.  The fields are 4 numbers followed by an unquoted string: the timestamp of position estimate, the wall clock at time of estimate, the horizon requested at estimation (set to -1 for offline trials), the slack time at estimation and, finally, the estimated position .

Returns code 405 in the nonstarted state, as no estimates exist.


### POST /TRIAL/estimates

Used for offline trials.  Must be called once at end of trial.

Each posted ASCII data line has the format `pts,pos` where `pts` is the position timestamp, a non-negative number and `pos` is the position estimate, a string whose format is trial-dependent.  A data line must be readable without error by the Python instruction `parse("{pts:f},{pos:S}", line)` from the `parse` library.

`Content-type` must be set to `text/csv; charset=us-ascii`.

Returns code 400 if the character set or character encoding is not as required; no estimates are set; the trial state does not change.

Returns code 405 if the trial is not running; returns the state data as with command `state`.

Returns code 409 if the server accepted only part of the estimates sent and the trial has finished normally.  Returns a message telling how many estimates were accepted, how many were rejected and the reason of the first failure.

Returns code 422 if not an offline trial or called before the `nexdtata` command; the trial state does not change.


### GET /TRIAL/log (xzcompr)

Returns the log file in plain text format or, if using `xzcompr` parameter, in xz compressed form.

Returns code 405 if no log file.


### GET /TRIAL/COMMAND

Returns code 404 if `TRIAL` does not exist.

Returns code 422 if `TRIAL` exists and `COMMAND` is not one of the above commands


<!-- Local Variables: -->
<!-- fill-column: 100 -->
<!-- End: -->
