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

## Making it better

The compiled persona is a decent start but it can't know things people never say in chat. Nobody types their own surname in their group chat, so the bot won't know it.

Put that stuff in `personas/janez/extra.md`. It gets stuck on the end of the prompt and, unlike `persona.md`, recompiling doesn't wipe it. Full name, age, job, family, whatever. Also good for fixing habits you don't like, mine kept opening every message with my name until I told it not to.

You can give them reaction gifs too:

```toml
[[reactions]]
name = "angry"
file = "angry.gif"
when = "Lev winds you up or is being mean"
```

Drop the file in `personas/janez/media/`. Tenor links work as `url = "..."` instead, which is better for those since Discord embeds them and they don't expire like CDN attachment links do.

The `when` line is what the model reads to decide, so be specific there.

## Cost

Compiling costs maybe a euro. After that it's per reply.

If you have a Claude Pro or Max subscription you can use it directly, no API key, set `provider = "claude-code"`. Your plan includes a monthly Agent SDK credit ($20 / $100 / $200 depending on tier) and this draws from that. When it runs out it just stops, it can't overcharge you unless you've specifically turned on extra usage credits.

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
