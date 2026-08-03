# mimicord

Makes a Discord bot that talks like one of your mates. You give it their chat history, it works out how they write, and then it sits in your server replying like them.

About 3000 of their messages is enough. The bot picks up the small stuff, whether they capitalise, whether they end sentences with a full stop, which slang they use, whether they fire off three short messages instead of one long one. That last one is what actually sells it.

## Setup

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/keksoslav/mimicord
cd mimicord
uv sync
cp .env.example .env
uv run mimicord new janez
```

Open `personas/janez/persona.toml` and fill in who you're copying (their Discord user id) and which model to use. Then run the pipeline:

```
uv run mimicord ingest janez --dce "path/to/exports"
uv run mimicord stats janez
uv run mimicord analyze janez     # this one takes a while
uv run mimicord compile janez
uv run mimicord index janez
```

`analyze` is the slow bit, it reads through the chat history in chunks and asks the model what this person is like. Half an hour for a big corpus. It caches as it goes so you can ctrl-c and pick it up later.

Then read `personas/janez/persona.md`. That file is the whole personality and it's worth actually reading, it's usually a bit uncanny. If it looks right:

```
uv run mimicord chat janez        # talk to them in your terminal
uv run mimicord run janez         # put them in Discord
```

There's also a GUI (`uv sync --extra gui` then `mimicord gui`) if you'd rather click through it. Same thing, buttons instead of commands.

You don't have to do any of the pipeline stuff, by the way. `mimicord new` writes a blank persona file you can just write by hand, and chat/run work fine with that.

## Getting the chat history

This is the annoying part.

The easy option is your own data export: Discord Settings, Privacy and Safety, Request all of my data. Takes a few days to arrive. Problem is it only has *your* messages, none of the replies, so it's not much use if you're copying someone else.

The better option is [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) pointed at a channel, exported as JSON. That gets everyone, with context, which makes a much better persona.

One warning. DCE will happily take your personal account token, and that's automating a user account, which is against Discord's ToS. They started actually enforcing it in early 2026 and people got logged out and warned. Use a bot token instead. mimicord itself never touches user tokens.

## The bot account

Go to the [developer portal](https://discord.com/developers/applications), new application, then Bot.

**Turn on Message Content Intent.** It's under Privileged Gateway Intents. If you forget, the bot connects fine and then just ignores everything, because every message arrives with empty content and there's nothing to reply to. Took me a while to work that one out.

Permissions: View Channels, Send Messages, Read Message History. Nothing else. Then invite it and run:

```
uv run mimicord doctor janez
```

which checks the token, the intent, and whether it's actually in a server, so you find out now rather than halfway through wondering why nothing happens.

Test with `--dry-run` first. It connects and listens for real but prints replies to the terminal instead of posting them.

**He ignores DMs.** Anyone who shares a server with the bot can open a DM with it, and that conversation ignores your channel allowlist, your always-on list and everything else scoped to a server. Nobody but the sender ever sees it. If your trigger keyword is the person's name, one bored mate can sit in a DM saying it over and over and spend your entire budget without anyone noticing. `allow_dms = false` is the default for that reason. There is no switch on Discord's side to stop bot DMs, so this has to be handled in the bot.

## Making it better

The compiled persona is a decent start but it can't know things people never say in chat. Nobody types their own surname in their group chat, so the bot won't know it.

Put that stuff in `personas/janez/extra.md`. It gets stuck on the end of the prompt and, unlike `persona.md`, recompiling doesn't wipe it. Full name, age, job, family, whatever. Also good for fixing habits you don't like, mine kept opening every message with my name until I told it not to.

He knows what day and time it is, machine local time, sent with every reply. Without it he invents whatever sounds plausible and gets caught: mine claimed he was knackered after a shift, on a Sunday, and somebody noticed. It rides in the live message rather than the system prompt, because a clock in a cached prefix would blow the cache every minute.

One thing to know: the `## Rules` block at the bottom of `persona.md` is owned by mimicord, not by you. The engine strips it and appends its own copy right at the end of the prompt, after `extra.md` and the reaction list, because whatever the model reads last carries the most weight and those are the lines that must not bend. Editing it in `persona.md` does nothing. `extra.md` is where your own rules go.

You can give them reaction gifs too:

```toml
[[reactions]]
name = "angry"
file = "angry.gif"
when = "Lev winds you up or is being mean"
```

Drop the file in `personas/janez/media/`. Tenor links work as `url = "..."` instead, which is better for those since Discord embeds them and they don't expire like CDN attachment links do.

The `when` line is what the model reads to decide, so be specific there.

## A pile of photos

Reactions are fine for a handful of gifs, but they don't scale: every one is a line in the prompt, and a persona prompt that is mostly a picture catalogue stops being a persona prompt. For an actual photo collection, don't put it in the prompt at all.

```
mkdir personas/janez/media/pictures      # name the files descriptively
uv run mimicord pictures janez           # index them, this is free and local
```

```toml
[pictures]
enabled = true
threshold = 1.1
```

He then sends one by writing `[pic: what he wants]` and the nearest caption is looked up on your machine. **The prompt cost is the same whether you have 20 pictures or 2,000**, because it never contains a list, just the one instruction and an auto-generated line of what subjects exist.

Captions come from the file and folder names, so `pictures/svit/squat.jpg` is findable as "svit squat". That means naming your files properly is the whole job. If a filename can't carry it, `captions.toml` beside `persona.toml` overrides any of them:

```toml
[captions]
"IMG_2481.jpg" = "timi na morju z ocali"
```

`mimicord pictures janez --ask "timi z pivom"` shows you what a description resolves to, and costs nothing, so tune with that rather than by talking to the bot.

Two things decide a match. The words have to actually overlap, compared in a way that survives Slovene endings so `pivom` finds `piva`, and only then does the embedding rank what's left. That order matters: doing it the other way round throws away a photo that plainly names the subject because it happened to rank twelfth. If nothing matches he sends nothing, which is the right failure.

For a small fixed set of reaction gifs, point a tag at a folder instead and it picks one at random each time:

```toml
[[reactions]]
name = "svit"
dir = "people/svit"
when = "photos of Svit, when he comes up or you're taking the piss out of him"
cooldown = 3600
```

This matters more than it sounds. I have 31 photos of my mates in there and as 31 separate entries they'd be 31 lines of instructions sitting in a prompt whose entire job is to make him sound like one person. Grouped into 9 folders it costs 861 characters and he still picks sensibly. A persona prompt that is mostly a picture catalogue stops being a persona prompt.

The other half is stopping him reaching for one every single message, which he absolutely will. Asking him not to does not work. `reaction_cooldown_seconds` under `[style]` is a floor between any two images, and `cooldown` on a single entry stops that specific one repeating. Mine are 4 minutes and an hour.

He can also start conversations instead of only answering them:

```toml
idle_hours = 24
idle_channels = ["123456789"]
```

If nobody has said anything in that channel for a day, he picks someone who was
talking there recently, pings them and says something. Leave `idle_channels` out
and it falls back to `always_on_channels`. It counts against the same hourly and
monthly caps as a normal reply, and the clock is wall clock, so restarting the
bot halfway through a quiet weekend does not reset it. Off by default.

He can see images too, which matters a lot if your chat is mostly memes and
screenshots. Install it with `uv sync --extra vision` and turn it on:

```toml
[vision]
enabled = true
max_images = 1
max_edge = 768
lookback = 2
```

Images are billed by pixel count, roughly `width * height / 750` tokens, so
`max_edge` is the whole cost story. Everything gets shrunk to that before it is
sent, which turns a 16,000 token phone photo into about 400. At 768 a typical
screenshot lands around 300 tokens, well under a cent, and only messages that
actually carry an image pay anything. Drop to 512 if you want it cheaper and
don't mind small text going blurry, go to 1024 if he keeps misreading
screenshots.

`lookback` is the other guard. It stops a meme from three messages ago being
re-sent with every single reply for the rest of the conversation, which is how
this gets expensive without you noticing.

He can't tell you who is in a photo, by the way. Claude won't identify real
people and you can't prompt around it.

## Attacking your own bot

Someone will try to break him. `mimicord redteam janez` gets there first:

```
uv run mimicord redteam janez
```

It runs a dozen attempts through the real engine and prints what he says to
each. Getting him to admit he is a bot, smuggling `ignore all previous
instructions` in through the chat, asking for his prompt back, faking a system
message, and making him write code or a bullet list. Some of it is checked
automatically, the rest you read, because no check can tell you whether he
sounded like himself.

Worth running before you change the prompt and again afterwards. Otherwise you
have no idea whether you improved it or just moved words around. Costs about a
dozen replies.

Someone in your server will eventually try to break him by pasting the same line
twenty times. Repeats get ignored, and a burst of messages gets one reply instead
of one each, because he waits `debounce_seconds` for you to finish typing before
he answers.

He also will not send the same message twice in a row, which is the thing that
makes a bot look like a bot. If he comes up with the same reply again he gets
told so and asked for another, once. If that one repeats too he says nothing,
which is at least something a person does.

## Cost

Compiling costs maybe a euro. After that it's per reply.

If you have a Claude Pro or Max subscription you can use it directly, no API key, set `provider = "claude-code"`. It goes through the Claude Code login on your machine and draws from your plan's usage limits.

Be aware that it shares those limits with your normal Claude use. Anthropic announced a separate monthly Agent SDK credit and then paused it, so right now a chatty bot eats into the same weekly allowance you use for everything else. Check `/usage` in a Claude Code session to see where you are. Running out stops the bot rather than charging you, unless you turned on usage credits yourself.

Opus works out around 7 cents a reply through that, so a couple of thousand replies a month on the top tier. Sonnet is cheaper and mostly fine, though if the person you're copying doesn't write in English then opus is noticeably better at it.

Otherwise: `anthropic` with an API key is cheaper per reply (no harness overhead), `deepseek` is cheapest, `ollama` is free if you're happy running it locally.

`mimicord inspect janez --cost` gives you an estimate. There's also `max_replies_per_hour` and `max_replies_per_month` in the config so it can't run away with your money while you're asleep.

## Don't be a dick about it

Ask the person first. Obviously.

Also tell the rest of the server. A bot with the BOT tag next to it is a joke everyone's in on, which is the fun bit. An account pretending to be a real person isn't, and it always comes out eventually.

Their messages stay on your machine. `personas/` is gitignored. The only thing that leaves is the prompt itself going to whichever model you picked.

## Tests

```
uv run pytest
```

All offline, no API keys needed.

## License

MIT
