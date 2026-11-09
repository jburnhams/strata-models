// A job's state as a word in its tone (a status is always a word, never a colour alone).
import { Status } from 'strata-react'
import type { JobStatus as S } from '../api'
import { STATUS_TONE, STATUS_WORD } from '../logic'

export default function JobStatus({ status }: { status: S }) {
  return <Status tone={STATUS_TONE[status]}>{STATUS_WORD[status]}</Status>
}
