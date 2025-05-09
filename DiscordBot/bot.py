import asyncio
import json
import logging
import pathlib
import re
from collections import defaultdict
from typing import Optional, List, Dict, Set

import discord
from discord.ext import commands

from report import Report, Violation, Priority, ModOutcome

# ────────────────────── TOKEN & LOGGING ──────────────────────────────
TOKENS_FILE = pathlib.Path(__file__).with_name("tokens.json")
try:
    TOKEN: str = json.loads(TOKENS_FILE.read_text())["discord"]
except Exception as exc:   # noqa: BLE001
    raise SystemExit(f"❌ Could not read Discord token from {TOKENS_FILE}") from exc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mod-bot")

# ─────────────────── GLOBAL IN-MEMORY STATE ─────────────────────────
REPORTS: Dict[int, List[Report]] = defaultdict(list)  # guild_id → [Report]
USER_VIOL_COUNTS: Dict[int, int] = defaultdict(int)   # offender_id → strikes
ACTIVE_DM_SESSIONS: Set[int]      = set()             # reporter IDs in flow

# ───────────────────── DISCORD BOILERPLATE ──────────────────────────
intents = discord.Intents.default()
intents.message_content = True               # need message text
bot = commands.Bot(command_prefix="!", intents=intents)

# ───────────────────────── HELPERS ───────────────────────────────────
def get_mod_channel(guild: discord.Guild,
                    main_channel: discord.TextChannel) -> Optional[discord.TextChannel]:
    """Return '<main>-mod' channel, or None if it doesn't exist."""
    if main_channel.name.endswith("-mod"):
        return main_channel
    return discord.utils.get(guild.text_channels,
                             name=f"{main_channel.name}-mod")


def yes_no(msg: discord.Message) -> Optional[bool]:
    lc = msg.content.lower().strip()
    if lc in {"yes", "y"}:
        return True
    if lc in {"no", "n"}:
        return False
    return None


async def prompt(dm: discord.DMChannel,
                 user: discord.User,
                 question: str,
                 check_func,
                 timeout: int = 120) -> Optional[discord.Message]:
    """Send *question*, wait for a reply that satisfies *check_func*."""
    await dm.send(question)
    try:
        return await bot.wait_for(
            "message",
            timeout=timeout,
            check=lambda m: m.author == user and m.channel == dm and check_func(m))
    except asyncio.TimeoutError:
        await dm.send("⏰ Timed out; cancelling the report.")
        return None


async def yn_prompt(dm: discord.DMChannel, user: discord.User, q: str) -> Optional[bool]:
    msg = await prompt(dm, user, q,
                       lambda m: yes_no(m) is not None or m.content.lower() == "cancel")
    if not msg or msg.content.lower() == "cancel":
        return None
    return yes_no(msg)


async def safe_dm(user: discord.User, content: str):
    """Try to DM; ignore Forbidden errors."""
    try:
        await user.send(content)
    except discord.Forbidden:
        pass


def mod_embed(report: Report,
              offending_msg: discord.Message,
              prior_reports: int) -> discord.Embed:
    colours = {
        Priority.EXTREME_URGENT: discord.Colour.red(),
        Priority.URGENT:         discord.Colour.orange(),
        Priority.NORMAL:         discord.Colour.blue(),
    }
    e = discord.Embed(
        title=f"{'🚨 ' if report.priority == Priority.EXTREME_URGENT else ''}"
              f"New report – {report.reason.value}",
        colour=colours[report.priority],
        description=offending_msg.content or "*[no text]*"
    )
    e.add_field(name="Author",   value=offending_msg.author.mention, inline=True)
    e.add_field(name="Reporter", value=f"<@{report.reporter_id}>",    inline=True)
    e.add_field(name="Message link", value=offending_msg.jump_url,    inline=False)
    if report.subcategory:
        e.add_field(name="Sub-category", value=report.subcategory, inline=True)
    if report.evidence_text:
        e.add_field(name="Extra context", value=report.evidence_text[:1024], inline=False)
    if report.attachment_urls:
        e.add_field(name="Attachments",
                    value="\n".join(report.attachment_urls)[:1024], inline=False)
    e.set_footer(text=f"Prior reports for user: {prior_reports}")
    return e

# ─────────────── MOD-ACTION BUTTON VIEW ──────────────────────────────
class ModActionView(discord.ui.View):
    def __init__(self, report: Report):
        super().__init__(timeout=None)
        self.rep = report

    # enforcement side-effects ----------------------------------------
    async def _apply_enforcement(self,
                                 outcome: ModOutcome,
                                 inter: discord.Interaction):
        guild   = inter.guild
        channel = guild.get_channel(self.rep.channel_id)
        offender = guild.get_member(self.rep.target_user_id) \
                   or await bot.fetch_user(self.rep.target_user_id)

        # delete offending message
        try:
            off_msg = await channel.fetch_message(self.rep.message_id)
            await off_msg.delete()
        except discord.NotFound:
            pass

        # DM offender
        if outcome == ModOutcome.WARN_USER:
            await safe_dm(offender,
                          f"⚠️ **Warning from {guild.name} moderators**\n\n"
                          "Your message was removed for violating sexual-content rules. "
                          "Please review the guidelines and avoid similar behaviour.")
        elif outcome == ModOutcome.SUSPEND_USER:
            await safe_dm(offender,
                          f"⛔ **Notice of suspension (test)** – {guild.name}\n\n"
                          "Due to severe or repeated violations, your posting privileges "
                          "are suspended. (This is a test environment; no real ban issued.)")

    # finalise ---------------------------------------------------------
    async def _resolve(self, inter: discord.Interaction, outcome: ModOutcome):
        if not self.rep.is_open:
            await inter.response.send_message("Already resolved.", ephemeral=True)
            return

        if outcome in {ModOutcome.REMOVE_MESSAGE,
                       ModOutcome.WARN_USER,
                       ModOutcome.SUSPEND_USER}:
            await self._apply_enforcement(outcome, inter)

        self.rep.close(outcome, inter.user.id)
        USER_VIOL_COUNTS[self.rep.target_user_id] += 1

        await inter.response.edit_message(
            content=f"✅ **{outcome.value}** – by {inter.user.mention}",
            view=None)

        if (outcome == ModOutcome.SUSPEND_USER
                and self.rep.reason in (Violation.GROOMING,
                                        Violation.SEXUAL_EXPLOITATION)):
            await inter.channel.send(
                f"⚠️ Evidence packet prepared for law enforcement "
                f"(report {self.rep.message_id}).")

    # buttons ----------------------------------------------------------
    @discord.ui.button(label="No violation", style=discord.ButtonStyle.secondary)
    async def _btn_no(self, inter, _): await self._resolve(inter, ModOutcome.NO_VIOLATION)

    @discord.ui.button(label="Remove", style=discord.ButtonStyle.danger)
    async def _btn_rm(self, inter, _): await self._resolve(inter, ModOutcome.REMOVE_MESSAGE)

    @discord.ui.button(label="Warn", style=discord.ButtonStyle.primary)
    async def _btn_warn(self, inter, _): await self._resolve(inter, ModOutcome.WARN_USER)

    @discord.ui.button(label="Suspend / Ban", style=discord.ButtonStyle.danger, emoji="⛔")
    async def _btn_ban(self, inter, _): await self._resolve(inter, ModOutcome.SUSPEND_USER)

# ─────────────────────── USER COMMAND (!report) ──────────────────────
@bot.command(name="report", help="!report <message_link> [reason]")
async def report_cmd(ctx: commands.Context,
                     message_link: str,
                     *,
                     reason: Optional[str] = None):
    m = re.search(r"discord\.com/channels/(\d+)/(\d+)/(\d+)", message_link)
    if not m:
        await ctx.reply("Invalid message link.")
        return
    gid, cid, mid = map(int, m.groups())

    guild   = bot.get_guild(gid)
    channel = guild.get_channel(cid)
    try:
        offending_msg = await channel.fetch_message(mid)
    except discord.NotFound:
        await ctx.reply("Message not found.")
        return

    # quick one-word reason path --------------------------------------
    if reason:
        viol = None
        rl = reason.lower()
        if "groom" in rl:
            viol = Violation.GROOMING
        elif "exploit" in rl:
            viol = Violation.SEXUAL_EXPLOITATION
        elif "harass" in rl:
            viol = Violation.SEXUAL_HARASSMENT
        else:
            await ctx.reply("Unknown reason. Use grooming / exploitation / harassment.")
            return
        await _file_simple_report(ctx.author, offending_msg, viol)
        await ctx.reply("✅ Report sent to moderators.")
        return

    # interactive DM flow ---------------------------------------------
    if ctx.author.id in ACTIVE_DM_SESSIONS:
        await ctx.reply("You already have an active reporting session in your DMs.")
        return

    dm = await ctx.author.create_dm()
    await ctx.reply("✉️ I just DM’d you some follow-up questions!")
    ACTIVE_DM_SESSIONS.add(ctx.author.id)
    try:
        rep = await _walk_user_flow(ctx.author, dm, offending_msg)
        if rep:
            prior = USER_VIOL_COUNTS[offending_msg.author.id]
            mod_channel = get_mod_channel(guild, channel)
            if mod_channel:
                await mod_channel.send(embed=mod_embed(rep, offending_msg, prior),
                                       view=ModActionView(rep))
            await dm.send("✅ Your report has been forwarded to the moderators. Thank you!")
    finally:
        ACTIVE_DM_SESSIONS.discard(ctx.author.id)

# ──────────────────── INTERACTIVE DM FLOW ────────────────────────────
async def _walk_user_flow(user: discord.User,
                          dm: discord.DMChannel,
                          offending_msg: discord.Message) -> Optional[Report]:
    # 1️⃣ choose main category
    q1 = ("Please select the **reason** for reporting this message:\n"
          "1️⃣ Sexual exploitation\n"
          "2️⃣ Grooming\n"
          "3️⃣ Sexual harassment\n"
          "_Type 1, 2, or 3 (or `cancel`)._")
    m1 = await prompt(dm, user, q1,
                      lambda x: x.content.lower().strip() in {"1", "2", "3", "cancel"})
    if not m1 or m1.content.lower() == "cancel":
        return None
    viol_map = {"1": Violation.SEXUAL_EXPLOITATION,
                "2": Violation.GROOMING,
                "3": Violation.SEXUAL_HARASSMENT}
    viol = viol_map[m1.content.strip()]
    subcat = ""

    # 2️⃣ branch-specific questions
    if viol == Violation.SEXUAL_EXPLOITATION:
        ok = await yn_prompt(dm, user,
                             "Does this message include content **for sexual purposes**? (yes/no)")
        if ok is None or not ok:
            await dm.send("Understood – please restart with a different category if needed.")
            return None
        m2 = await prompt(dm, user,
                          "Which best describes the exploitation?\n"
                          "1️⃣ Coercion\n2️⃣ Manipulation\n3️⃣ Enticement",
                          lambda x: x.content.strip() in {"1", "2", "3"})
        if not m2:
            return None
        subcat = {"1": "Coercion", "2": "Manipulation", "3": "Enticement"}[m2.content.strip()]

    elif viol == Violation.GROOMING:
        ok = await yn_prompt(dm, user,
                             "Do you suspect this user is building an "
                             "**inappropriate relationship with a minor**? (yes/no)")
        if ok is None or not ok:
            await dm.send("Got it – you can restart with a different category.")
            return None
        subcat = "Minor-targeted grooming"

    elif viol == Violation.SEXUAL_HARASSMENT:
        m2 = await prompt(dm, user,
                          "What kind of harassment are you reporting?\n"
                          "1️⃣ Unwanted sexual messages\n"
                          "2️⃣ Repeated advances after rejection\n"
                          "3️⃣ Inappropriate images or memes",
                          lambda x: x.content.strip() in {"1", "2", "3"})
        if not m2:
            return None
        subcat = {"1": "Unwanted sexual messages",
                  "2": "Repeated advances after rejection",
                  "3": "Inappropriate images/memes"}[m2.content.strip()]

    # 3️⃣ evidence upload
    add_ev = await yn_prompt(dm, user,
                             "Would you like to **attach additional messages or screenshots** "
                             "as evidence? (yes/no)")
    evidence_text = ""
    attach_urls: List[str] = []
    if add_ev:
        await dm.send("Send any extra context now. Type `done` when finished.")
        while True:
            m = await bot.wait_for("message",
                                    timeout=180,
                                    check=lambda x: x.author == user and x.channel == dm)
            if m.content.lower().startswith("done"):
                break
            evidence_text += m.content + "\n"
            attach_urls.extend(att.url for att in m.attachments)

    # 4️⃣ thank-you & optional block
    await dm.send("Thank you for your report. Our moderators will review it shortly.")
    wants_block = False
    blk = await yn_prompt(dm, user,
                          "Would you like to **block this user** from contacting you? (yes/no)")
    if blk:
        wants_block = True
        await dm.send("To block: right-click the user’s name → *Block*. Stay safe! ❤️")

    # 5️⃣ create & store report
    rep = Report(
        reporter_id          = user.id,
        guild_id             = offending_msg.guild.id,
        channel_id           = offending_msg.channel.id,
        message_id           = offending_msg.id,
        target_user_id       = offending_msg.author.id,
        reason               = viol,
        subcategory          = subcat,
        evidence_text        = evidence_text.strip(),
        attachment_urls      = attach_urls,
        reporter_wants_block = wants_block
    )
    REPORTS[rep.guild_id].append(rep)
    return rep

# ───────────────── SIMPLE ONE-WORD REPORT ────────────────────────────
async def _file_simple_report(reporter: discord.User,
                              offending_msg: discord.Message,
                              viol: Violation):
    guild = offending_msg.guild
    rep = Report(
        reporter_id    = reporter.id,
        guild_id       = guild.id,
        channel_id     = offending_msg.channel.id,
        message_id     = offending_msg.id,
        target_user_id = offending_msg.author.id,
        reason         = viol
    )
    REPORTS[guild.id].append(rep)
    prior = USER_VIOL_COUNTS[offending_msg.author.id]
    mod_channel = get_mod_channel(guild, offending_msg.channel)
    if mod_channel:
        await mod_channel.send(embed=mod_embed(rep, offending_msg, prior),
                               view=ModActionView(rep))

# ─────────────────────── LIFECYCLE (resume) ──────────────────────────
@bot.event
async def on_ready():
    log.info("Logged in as %s (%s)", bot.user, bot.user.id)
    # re-attach views after restart
    for gid, reps in REPORTS.items():
        guild = bot.get_guild(gid)
        if not guild:
            continue
        sample_main = discord.utils.find(
            lambda c: c.name.startswith("group-") and not c.name.endswith("-mod"),
            guild.text_channels)
        mod_chan = get_mod_channel(guild, sample_main) if sample_main else None
        if not mod_chan:
            continue
        for rep in reps:
            if rep.is_open:
                try:
                    msg = await mod_chan.fetch_message(rep.message_id)
                    await msg.edit(view=ModActionView(rep))
                except discord.NotFound:
                    pass

# ─────────────────────────── RUN BOT ─────────────────────────────────
if __name__ == "__main__":
    bot.run(TOKEN)
